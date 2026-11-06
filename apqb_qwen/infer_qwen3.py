"""Inference with a Full APQB Bloch Filter patched into a Qwen3 model.

Runs the same prompts through the plain model (baseline) and the filtered model,
and reports the logit drift, perplexity on a reference text, and generations.

  python infer_qwen3.py                                   # Qwen/Qwen3-0.6B on CPU, untrained filter
  python infer_qwen3.py --weights apqb_qwen38.pt --model Qwen/Qwen3.8-27B   # trained filter (GPU)

Without --weights the filter parameters are at their initial values, so the
result shows only what the blend does untrained; lam_init sets the blend
(sigmoid(-3) ~ 5%, the train_qwen38.py default).
"""
import argparse, math, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from apqb_filter import patch_model, patch_model_control, APQBGatedMLP, AffineSiLUMLP

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
ap.add_argument("--weights", default=None, help="filter state saved by train_qwen38.py")
ap.add_argument("--lam_init", type=float, default=-3.0)
ap.add_argument("--max_new_tokens", type=int, default=80)
ap.add_argument("--prompt", action="append")
ap.add_argument("--no_baseline", action="store_true")
ap.add_argument("--control", action="store_true", help="patch the affine-SiLU control instead of the APQB filter")
a = ap.parse_args()

prompts = a.prompt or ["量子ビットとは何ですか？簡潔に説明してください。",
                       "Explain in two sentences why the sky is blue."]
ref_text = ("東京は日本の首都であり、政治・経済・文化の中心地である。"
            "The quick brown fox jumps over the lazy dog. Water boils at 100 degrees Celsius at sea level.")

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if dev == "cuda" else torch.float32
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, device_map="auto" if dev == "cuda" else None).eval()


def chat_ids(p):
    text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    return tok(text, return_tensors="pt").input_ids.to(model.device)


@torch.no_grad()
def run(tag):
    ids = tok(ref_text, return_tensors="pt").input_ids.to(model.device)
    out = model(ids, labels=ids)
    print(f"[{tag}] perplexity on reference text: {math.exp(out.loss.item()):.3f}")
    logits, gens = [], []
    for p in prompts:
        x = chat_ids(p)
        logits.append(model(x).logits[0, -1].float().cpu())
        t = time.time()
        y = model.generate(x, max_new_tokens=a.max_new_tokens, do_sample=False)
        s = tok.decode(y[0, x.shape[1]:], skip_special_tokens=True).strip()
        gens.append(s)
        print(f"[{tag}] ({time.time()-t:.1f}s) Q: {p}\n  A: {s}\n")
    return logits, gens


base = None if a.no_baseline else run("baseline")

n = patch_model_control(model) if a.control else patch_model(model, lam_init=a.lam_init)
tag = "control" if a.control else "APQB-filtered"
for m in model.modules():
    if isinstance(m, (APQBGatedMLP, AffineSiLUMLP)):
        m.to(model.device)
        for p in m.filt.parameters(): p.data = p.data.float()
    if isinstance(m, APQBGatedMLP):
        m.lam_raw.data, m.out_scale.data = m.lam_raw.data.float(), m.out_scale.data.float()
if a.weights:
    state = torch.load(a.weights, map_location="cpu")
    missing = model.load_state_dict(state, strict=False)
    print(f"loaded {len(state)} filter tensors; unexpected={len(missing.unexpected_keys)}")
lams = [torch.sigmoid(m.lam_raw).flatten() for m in model.modules() if isinstance(m, APQBGatedMLP)]
lam = torch.cat(lams).mean().item() if lams else float("nan")
print(f"patched MLPs: {n} ({tag}), mean lambda = {lam:.4f}\n")

filt = run(tag)

if base:
    for p, lb, lf, gb, gf in zip(prompts, base[0], filt[0], base[1], filt[1]):
        pb, pf = lb.log_softmax(-1), lf.log_softmax(-1)
        kl = (pb.exp() * (pb - pf)).sum().item()
        print(f"next-token KL(base||filtered)={kl:.4f}  top1 same={lb.argmax()==lf.argmax()}  "
              f"generation identical={gb==gf}  :: {p}")
