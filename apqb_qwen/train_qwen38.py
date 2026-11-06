"""Attach Full APQB Bloch Filters to Qwen3.8-27B and train only the filter params (GPU required).

  pip install torch transformers datasets accelerate
  python train_qwen38.py --data my_text.txt --steps 2000 --out apqb_qwen38.pt
Compare against baseline with --baseline (same budget, lam frozen at 0 => untouched model).
"""
import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from apqb_filter import patch_model, apqb_parameters, freeze_except_apqb

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
ap.add_argument("--data", required=True)          # plain-text file
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--seq", type=int, default=1024)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--lam_init", type=float, default=-3.0)   # sigmoid(-3)=5% blend at start
ap.add_argument("--out", default="apqb_qwen38.pt")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto")
print("patched MLPs:", patch_model(model, lam_init=a.lam_init))
freeze_except_apqb(model)
params = apqb_parameters(model)
for p in params: p.data = p.data.float()           # keep filter params in fp32
print("trainable params:", sum(p.numel() for p in params))
model.gradient_checkpointing_enable(); model.enable_input_require_grads()

ids = tok(open(a.data, encoding="utf-8").read(), return_tensors="pt").input_ids[0]
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
for step in range(a.steps):
    i = torch.randint(0, len(ids) - a.seq - 1, (1,)).item()
    x = ids[i:i + a.seq + 1].unsqueeze(0).to(model.device)
    loss = model(x[:, :-1], labels=x[:, 1:]).loss
    loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step(); opt.zero_grad()
    if step % 50 == 0: print(step, round(loss.item(), 4), flush=True)
torch.save({n: p.detach().cpu() for n, p in model.named_parameters()
            if any(k in n for k in ("filt.", "lam_raw", "out_scale"))}, a.out)
