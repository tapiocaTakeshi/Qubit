"""Sanity tests: filter math, identity-at-init, and a short training run on a tiny Qwen3."""
import torch, math
from transformers import Qwen3Config, Qwen3ForCausalLM
from apqb_filter import *

torch.manual_seed(0)
# 1) paper's example: n=2, kappa=1, X-direction: phi=0 -> 1, phi=pi -> 0
X = (math.pi/2, 0.0)
f = lambda ph: bloch_filter(torch.tensor(2.), torch.tensor(ph), torch.tensor(1.), torch.tensor(X[0]), torch.tensor(X[1])).item()
print("F(phi=0)=%.3f  F(phi=pi)=%.3f" % (f(0.), f(math.pi)))
assert abs(f(0.) - 1) < 1e-4 and abs(f(math.pi)) < 1e-4
# range check
n = torch.rand(10000)*4; ph = torch.rand(10000)*6.28; k = torch.rand(10000)
a = torch.rand(10000)*3.14; b = torch.rand(10000)*6.28
F = bloch_filter(n, ph, k, a, b); print("range", F.min().item(), F.max().item())
assert F.min() >= -1e-5 and F.max() <= 1+1e-5

# 2) tiny Qwen3, identity at init
cfg = Qwen3Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=128)
m = Qwen3ForCausalLM(cfg).eval()
ids = torch.randint(0, 256, (2, 32))
with torch.no_grad(): ref = m(ids).logits
print("patched layers:", patch_model(m))
with torch.no_grad(): out = m(ids).logits
print("max diff at init: %.2e" % (out-ref).abs().max())
assert (out-ref).abs().max() < 1e-3

# 3) train only APQB params on a copy task; loss must fall, lam must grow
freeze_except_apqb(m); m.train()
params = apqb_parameters(m); print("trainable:", sum(p.numel() for p in params))
opt = torch.optim.AdamW(params, lr=5e-2)
data = (torch.arange(32).repeat(8,1) * 7 % 256)
for step in range(60):
    loss = m(data, labels=data).loss
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 20 == 0 or step == 59: print(step, round(loss.item(), 4))
lam = torch.sigmoid(torch.cat([p.flatten() for n_, p in m.named_parameters() if "lam_raw" in n_])).mean().item()
print("mean lambda after training: %.4f" % lam)
