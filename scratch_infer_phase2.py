#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig, NeuroQuantumTokenizer

ARCH = dict(embed_dim=544, hidden_dim=1280, num_heads=8, num_layers=7, max_seq_len=1024, vocab_size=32000)
CKPT = "qbnn100m_v2_phase2_best.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = NeuroQuantumTokenizer(vocab_size=32000, model_file="neuroq_tokenizer.model")
config = NeuroQuantumConfig(
    vocab_size=ARCH["vocab_size"], embed_dim=ARCH["embed_dim"], hidden_dim=ARCH["hidden_dim"],
    num_heads=ARCH["num_heads"], num_layers=ARCH["num_layers"], max_seq_len=ARCH["max_seq_len"],
    dropout=0.0, lambda_entangle=0.0,
)
model = NeuroQuantum(config=config, tokenizer=tokenizer).to(device)
state = torch.load(CKPT, map_location=device, weights_only=False)
model.load_state_dict(state, strict=False)
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
print(f"チェックポイント: {CKPT}")
print("=" * 80)
for p in prompts:
    print(f"\n[プロンプト] {p}")
    print(f"[生成結果] {generate(p)}")
