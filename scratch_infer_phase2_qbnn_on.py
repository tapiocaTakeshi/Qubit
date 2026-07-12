#!/usr/bin/env python3
"""J/θ(QBNNのもつれ強度λ)を学習時の上書き(λ=0付近)から
config本来の値(lambda_entangle=0.5 -> 各層 lambda_min=0.25, lambda_max=0.75)に戻して推論する。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig, NeuroQuantumTokenizer, QBNNLayer

ARCH = dict(embed_dim=544, hidden_dim=1280, num_heads=8, num_layers=7, max_seq_len=1024, vocab_size=32000)
CKPT = "qbnn100m_v2_phase2_best.pt"
LAMBDA_ENTANGLE = 0.5  # config本来のデフォルト

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = NeuroQuantumTokenizer(vocab_size=32000, model_file="neuroq_tokenizer.model")
config = NeuroQuantumConfig(
    vocab_size=ARCH["vocab_size"], embed_dim=ARCH["embed_dim"], hidden_dim=ARCH["hidden_dim"],
    num_heads=ARCH["num_heads"], num_layers=ARCH["num_layers"], max_seq_len=ARCH["max_seq_len"],
    dropout=0.0, lambda_entangle=LAMBDA_ENTANGLE,
)
model = NeuroQuantum(config=config, tokenizer=tokenizer).to(device)
state = torch.load(CKPT, map_location=device, weights_only=False)
model.load_state_dict(state, strict=False)

# J/θ(lambda_min, lambda_max)をconfig本来の値に戻す
n_restored = 0
for m in model.modules():
    if isinstance(m, QBNNLayer):
        m.lambda_min = LAMBDA_ENTANGLE * 0.5
        m.lambda_max = LAMBDA_ENTANGLE * 1.5
        n_restored += 1
print(f"✓ QBNNLayer {n_restored}個の λ を元に戻しました (lambda_min={LAMBDA_ENTANGLE*0.5}, lambda_max={LAMBDA_ENTANGLE*1.5})")

model.eval()

prompts = [
    "日本の首都は",
    "人工知能とは、",
    "今日はいい天気なので、",
]

@torch.no_grad()
def generate(prompt, max_new_tokens=60, temperature=0.8, top_k=40):
    ids = [tokenizer.bof_id, tokenizer.bos_id] + tokenizer.encode(prompt, add_special=False)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        if input_ids.shape[1] >= ARCH["max_seq_len"]:
            break
        logits = model(input_ids)
        next_logits = logits[0, -1, :] / temperature
        topk_vals, topk_idx = torch.topk(next_logits, top_k)
        probs = torch.softmax(topk_vals, dim=-1)
        next_id = topk_idx[torch.multinomial(probs, 1)].item()
        if next_id in (tokenizer.eos_id, tokenizer.eof_id):
            break
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)
    out_ids = input_ids[0].tolist()
    return tokenizer.decode(out_ids)

print("=" * 80)
print(f"チェックポイント: {CKPT} (QBNN λ 有効化)")
print("=" * 80)
for p in prompts:
    print(f"\n[プロンプト] {p}")
    print(f"[生成結果] {generate(p)}")
