"""Attach Full APQB Bloch Filters to a Qwen3 model and train only the filter params.

  pip install torch transformers datasets accelerate
  python train_qwen38.py --data train.txt --eval_data valid.txt --steps 2000 --out apqb_qwen38.pt
  python train_qwen38.py --model Qwen/Qwen3-0.6B --dtype float32 --seq 256 ...   # CPU-sized run

With --eval_data, held-out perplexity is reported for the untouched base model
(baseline), the patched model before training, and the patched model after training.
"""
import argparse, math, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from apqb_filter import (patch_model, patch_model_control, apqb_parameters, freeze_except_apqb,
                         APQBGatedMLP, AffineSiLUMLP)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
ap.add_argument("--data", required=True)          # plain-text file
ap.add_argument("--eval_data", default=None)      # held-out plain-text file
ap.add_argument("--eval_windows", type=int, default=16)
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--seq", type=int, default=1024)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--lam_init", type=float, default=-3.0)   # sigmoid(-3)=5% blend at start
ap.add_argument("--dtype", default="bfloat16")
ap.add_argument("--log_every", type=int, default=50)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--control", action="store_true",
                help="train a per-channel affine on SiLU instead of the APQB filter (baseline)")
ap.add_argument("--out", default="apqb_qwen38.pt")
a = ap.parse_args()
torch.manual_seed(a.seed)

cuda = torch.cuda.is_available()
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype),
                                             device_map="auto" if cuda else None)
dev = model.device


def load_ids(path):
    return tok(open(path, encoding="utf-8").read(), return_tensors="pt").input_ids[0]


eval_ids = load_ids(a.eval_data) if a.eval_data else None


@torch.no_grad()
def eval_ppl():
    """Mean token NLL over fixed, evenly spaced held-out windows."""
    model.eval()
    stride = (len(eval_ids) - a.seq) // a.eval_windows
    nll = 0.0
    for k in range(a.eval_windows):
        x = eval_ids[k * stride:k * stride + a.seq].unsqueeze(0).to(dev)
        nll += model(x, labels=x).loss.item()   # HF shifts labels internally
    model.train()
    return math.exp(nll / a.eval_windows)


if eval_ids is not None:
    print(f"held-out ppl, base model (baseline): {eval_ppl():.3f}", flush=True)

n_patched = patch_model_control(model) if a.control else patch_model(model, lam_init=a.lam_init)
print("patched MLPs:", n_patched, "(control: affine SiLU)" if a.control else "(APQB filter)")
for m in model.modules():
    if isinstance(m, (APQBGatedMLP, AffineSiLUMLP)):
        m.to(dev)
freeze_except_apqb(model)
params = apqb_parameters(model)
for p in params: p.data = p.data.float()           # keep filter params in fp32
print("trainable params:", sum(p.numel() for p in params))
if eval_ids is not None:
    print(f"held-out ppl, patched before training: {eval_ppl():.3f}", flush=True)
model.gradient_checkpointing_enable(); model.enable_input_require_grads()
model.train()

ids = load_ids(a.data)
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
data_rng = torch.Generator().manual_seed(a.seed)   # same windows for filter and control runs
t0, run = time.time(), 0.0
for step in range(a.steps):
    i = torch.randint(0, len(ids) - a.seq, (1,), generator=data_rng).item()
    x = ids[i:i + a.seq].unsqueeze(0).to(dev)
    loss = model(x, labels=x).loss                  # HF shifts labels internally
    loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step(); opt.zero_grad()
    run += loss.item()
    if (step + 1) % a.log_every == 0:
        lams = [torch.sigmoid(m.lam_raw).flatten() for m in model.modules() if isinstance(m, APQBGatedMLP)]
        lam = torch.cat(lams).mean().item() if lams else float("nan")
        print(f"step {step+1}  loss {run/a.log_every:.4f}  mean_lambda {lam:.4f}  "
              f"{time.time()-t0:.0f}s", flush=True)
        run = 0.0

if eval_ids is not None:
    print(f"held-out ppl, patched after training: {eval_ppl():.3f}", flush=True)
torch.save({n: p.detach().cpu() for n, p in model.named_parameters()
            if any(k in n for k in ("filt.", "lam_raw", "out_scale"))}, a.out)
print("saved", a.out)
