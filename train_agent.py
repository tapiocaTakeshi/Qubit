"""Bounded agent SFT with the runtime prompts, answer-only loss and held-out tasks.

Validation is the default; --train explicitly trains a separate candidate. This
never replaces/syncs the serving checkpoint and never downloads a model/dataset.
External JSONL rows must use the same request/observations/target contract.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

from neuroquantum_agent import calculate, validate_request
from neuroquantum_agent_protocol import answer_prompt, decision_prompt, parse_call, strict_json


def bootstrap_records():
    """Synthetic curriculum, not a claim of broad conversation competence."""
    records = []

    def add(task, target, *, stage="decision", history=None, observations=None,
            tools=None, documents=None, group=None):
        records.append({"group": group or task, "request": {"prompt": task, "parameters": {
            "protocol": 2, "history": history or [], "documents": documents or []}},
            "tools": tools or ["calculator"], "observations": observations or [],
            "stage": stage, "target": target})

    finish = {"name": "finish", "arguments": {}}
    for a in range(2, 22):
        for b in (3, 7, 11):
            expression = f"{a}*{b}"
            task = f"{a}個入りの商品を{b}箱買いました。全部で何個ですか？"
            call = {"name": "calculator", "arguments": {"expression": expression}}
            observation = {"tool": "calculator", "input": expression, "arguments": call["arguments"],
                "call_id": "call_1", "status": "completed",
                "output": json.dumps(calculate(expression), ensure_ascii=False)}
            add(task, call)
            add(task, finish, observations=[observation])
            add(task, f"全部で{a * b}個です。", stage="answer", observations=[observation])
    conversations = [
        ("君ユーモアのセンスあるね", [], "ありがとう。そう言ってもらえてうれしいです。"),
        ("え？無視？", [{"role": "user", "content": "君ユーモアのセンスあるね"},
          {"role": "assistant", "content": "映画を選びますか？"}],
          "ごめんなさい。話がそれました。褒めてくれてありがとう。"),
        ("こんにちは", [], "こんにちは。何をお手伝いしましょうか？"),
        ("英語で挨拶して", [], "Hello! How can I help you?"),
        ("ファイルを削除して", [], "このエージェントにはファイルを削除する機能がありません。"),
        ("分からないときはどうする？", [], "確認できないことは断定せず、必要な情報を尋ねます。"),
        ("私の名前は？", [], "まだお名前を伺っていないため、分かりません。"),
        ("私の名前は？", [{"role": "user", "content": "名前は空です"}], "空さんです。"),
    ]
    for task, history, answer in conversations:
        add(task, finish, history=history)
        add(task, answer, stage="answer", history=history)
    for task, question in [("それを計算して", "計算したい式や数値を教えてください。"),
                           ("比較して", "何と何を、どの観点で比較しますか？")]:
        add(task, {"name": "clarify", "arguments": {"question": question}})
    # Grounded examples use explicitly synthetic supplied sources, never invented live facts.
    for i in range(12):
        task = f"資料{i}によると受付は何時ですか？"
        document = f"資料{i}: 受付は{i + 8}時です。"
        query = f"資料{i} 受付"
        call = {"name": "document_search", "arguments": {"query": query}}
        observation = {"tool": "document_search", "input": query, "arguments": call["arguments"],
            "call_id": "call_1", "status": "completed",
            "output": json.dumps([{"doc_id": 0, "text": document}], ensure_ascii=False)}
        args = {"tools": ["calculator", "document_search"], "documents": [document]}
        add(task, call, **args)
        add(task, finish, observations=[observation], **args)
        add(task, f"資料{i}では{i + 8}時とされています。", stage="answer", observations=[observation], **args)
    task = "検索で調べて"
    query = "指定された話題"
    call = {"name": "web_search", "arguments": {"query": query}}
    for failed in (True, False):
        obs = {"tool": "web_search", "input": query, "arguments": call["arguments"],
            "call_id": "call_1", "status": "failed" if failed else "completed",
            "output": "処理に失敗しました。" if failed else "[]"}
        add(task, finish, tools=["calculator", "web_search"], observations=[obs])
        add(task, "検索から根拠を取得できませんでした。", stage="answer",
            tools=["calculator", "web_search"], observations=[obs])
    # Include a concrete query and preserve it exactly in the call.
    add("QubitのドキュメントをWeb検索して", {"name": "web_search", "arguments": {
        "query": "Qubit ドキュメント"}}, tools=["calculator", "web_search"])
    return records


def compile_record(row):
    request = row["request"]
    task, _, history, _ = validate_request(request)
    tools = row["tools"]
    if (not isinstance(tools, list) or not tools or len(set(tools)) != len(tools)
            or any(t not in ("calculator", "web_search", "document_search") for t in tools)):
        raise ValueError("Invalid training tools")
    choice = request.get("parameters", {}).get("tool_choice", "auto")
    if choice == "none":
        tools = []
    elif choice not in ("auto", "required"):
        if choice not in tools:
            raise ValueError("Unavailable training tool_choice")
        tools = [choice]
    observations = row.get("observations", [])
    if not isinstance(observations, list) or len(observations) > 4:
        raise ValueError("Invalid observations")
    for obs in observations:
        if (not isinstance(obs, dict) or obs.get("tool") not in tools
                or obs.get("status") not in ("completed", "failed")
                or not isinstance(obs.get("output"), str) or len(obs["output"]) > 4000):
            raise ValueError("Invalid observation")
        parse_call(json.dumps({"name": obs["tool"], "arguments": obs["arguments"]}), tools)
    if row["stage"] == "decision":
        target = json.dumps(row["target"], ensure_ascii=False, separators=(",", ":"))
        parse_call(target, tools)
        prompt = decision_prompt(task, history, tools, observations, choice)
    elif row["stage"] == "answer":
        target = row["target"]
        if not isinstance(target, str) or not 0 < len(target.strip()) <= 4000:
            raise ValueError("Invalid answer")
        prompt = answer_prompt(task, history, observations)
    else:
        raise ValueError("Invalid stage")
    return prompt, target


def split_records(records):
    """Keep every stage of a task in the same split; no post-split replication."""
    train, validation, seen = [], [], set()
    for row in records:
        prompt, target = compile_record(row)
        identity = (prompt, target)
        if identity in seen:
            continue
        seen.add(identity)
        # Task text, not an arbitrary row ID, defines the leakage boundary.
        group = row["request"]["prompt"]
        bucket = int(hashlib.sha256(group.encode()).hexdigest(), 16) % 5
        (validation if bucket == 0 else train).append(row)
    if not train or not validation:
        raise ValueError("Need distinct training and validation task groups")
    return train, validation


def encode_record(row, tokenizer, max_length):
    prompt, answer = compile_record(row)
    prefix = [tokenizer.bof_id, tokenizer.bos_id] + tokenizer.encode(
        f"質問: {prompt}\n回答:", add_special=False)
    target = tokenizer.encode(answer, add_special=False) + [tokenizer.eos_id, tokenizer.eof_id]
    if len(prefix) + len(target) > max_length:
        raise ValueError("Training example exceeds context window; do not silently truncate")
    return prefix + target, [-100] * len(prefix) + target


def train_candidate(handler, train, validation, *, epochs, max_steps, lr):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(42)
    encoded = [encode_record(r, handler.tokenizer, handler.config["max_seq_len"]) for r in train]
    heldout = [encode_record(r, handler.tokenizer, handler.config["max_seq_len"]) for r in validation]
    optimizer = torch.optim.AdamW(handler.model.parameters(), lr=lr)
    stats = {"processed_samples": 0, "target_tokens": 0, "optimizer_steps": 0}

    def loss_for(item):
        ids, labels = item
        x = torch.tensor([ids], dtype=torch.long, device=handler.device)
        y = torch.tensor(labels[1:], dtype=torch.long, device=handler.device)
        logits = handler.model(x)[0, :-1, :]
        return F.cross_entropy(logits, y, ignore_index=-100), int((y != -100).sum())

    def evaluate():
        handler.model.eval()
        total, count = 0., 0
        with torch.no_grad():
            for item in heldout:
                loss, tokens = loss_for(item)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite validation loss")
                total += loss.item() * tokens
                count += tokens
        return total / count

    stats["validation_loss_before"] = evaluate()
    rng = random.Random(42)
    try:
        for _ in range(epochs):
            rng.shuffle(encoded)
            handler.model.train()
            for item in encoded:
                if stats["optimizer_steps"] >= max_steps:
                    break
                optimizer.zero_grad(set_to_none=True)
                loss, tokens = loss_for(item)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss; candidate not saved")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(handler.model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                stats["optimizer_steps"] += 1
                stats["processed_samples"] += 1
                stats["target_tokens"] += tokens
            if stats["optimizer_steps"] >= max_steps:
                break
        stats["validation_loss_after"] = evaluate()
    finally:
        handler.model.eval()
    return stats


def load_training_model(directory):
    """Read only the explicitly supplied directory; never resize/reinitialize weights."""
    import torch
    from types import SimpleNamespace
    from neuroquantum_layered import (NeuroQuantum, NeuroQuantumConfig,
                                      NeuroQuantumTokenizer, migrate_legacy_state_dict)
    directory = directory.resolve(strict=True)
    checkpoint = next((directory / name for name in
        ("qbnn_checkpoint.pt", "neuroq_checkpoint.pt", "checkpoint.pt", "model.pt")
        if (directory / name).is_file()), None)
    tokenizer_path = directory / "neuroq_tokenizer.model"
    if checkpoint is None or not tokenizer_path.is_file():
        raise ValueError("Explicit model directory needs a checkpoint and matching neuroq_tokenizer.model")
    source = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = dict(source["config"])
    tokenizer = NeuroQuantumTokenizer(vocab_size=config["vocab_size"], model_file=str(tokenizer_path))
    if tokenizer.sp is None or tokenizer.actual_vocab_size != config["vocab_size"]:
        raise ValueError("Tokenizer vocabulary mismatch; refusing to resize learned weights")
    model = NeuroQuantum(config=NeuroQuantumConfig(
        vocab_size=config["vocab_size"], embed_dim=config["embed_dim"],
        hidden_dim=config.get("hidden_dim", config["embed_dim"] * 2),
        num_heads=config["num_heads"], num_layers=config["num_layers"],
        max_seq_len=config["max_seq_len"], dropout=config.get("dropout", .1),
        lambda_entangle=config.get("entangle_strength", .5)))
    model.load_state_dict(migrate_legacy_state_dict(source["model_state"], model), strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return SimpleNamespace(model=model.to(device), tokenizer=tokenizer, config=config,
                           device=device, ckpt_path=str(checkpoint))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="Optional normalized JSONL; default synthetic curriculum")
    parser.add_argument("--output-dir", type=Path, required=True, help="Must not already exist")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 3 or not 1 <= args.max_steps <= 200 or not 0 < args.lr <= 1e-4:
        parser.error("epochs 1..3, max-steps 1..200 and lr (0, 1e-4] required")
    if args.dataset:
        if args.dataset.stat().st_size > 20_000_000:
            parser.error("Dataset exceeds 20 MB")
        records = [strict_json(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        records = bootstrap_records()
    if len(records) > 10000:
        parser.error("Dataset exceeds 10000 records")
    train, validation = split_records(records)
    report = {"trained": False, "source": str(args.dataset or "synthetic-bootstrap-v1"),
              "train_records": len(train), "validation_records": len(validation),
              "processed_samples": 0, "protocol": 2, "seed": 42}
    if args.train and not args.model_dir:
        parser.error("--train requires --model-dir containing the existing checkpoint and tokenizer")
    # Exclusive directory creation prevents accidental overwriting of model artifacts.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, rows in (("train", train), ("validation", validation)):
        with (args.output_dir / f"{name}.jsonl").open("x", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.train:
        import torch
        handler = load_training_model(args.model_dir)
        report.update(train_candidate(handler, train, validation,
            epochs=args.epochs, max_steps=args.max_steps, lr=args.lr))
        report["trained"] = True
        report["tokenizer_sha256"] = hashlib.sha256(
            (args.model_dir / "neuroq_tokenizer.model").read_bytes()).hexdigest()
        torch.save({"model_state": handler.model.state_dict(), "config": handler.config,
                    "agent_training": report, "source_checkpoint": handler.ckpt_path},
                   args.output_dir / "agent_candidate.pt")
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
