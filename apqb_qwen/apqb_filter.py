"""Full APQB Bloch Filter and its integration into Qwen-style gated MLPs.

F(b,u) = (1 + b.u)/2,  b = (k*sqrt(n(4-n))/2*cos(phi), k*sqrt(n(4-n))/2*sin(phi), 1-n/2)
u(alpha,beta) = (sin a cos b, sin a sin b, cos a)
=> F = 1/2 [1 + (1-n/2) cos a + k*sqrt(n(4-n))/2 * sin a * cos(phi-beta)]
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as Fn


def bloch_filter(n, phi, kappa, alpha, beta, eps=1e-6):
    """All args broadcastable. n in [0,4]. Returns F in [0,1]."""
    coh = kappa * torch.sqrt((n * (4 - n)).clamp_min(eps)) / 2
    z = 1 - n / 2
    return 0.5 * (1 + z * torch.cos(alpha) + coh * torch.sin(alpha) * torch.cos(phi - beta))


class FullAPQBFilter(nn.Module):
    """Per-channel Bloch filter driven by two streams.

    s   -> n   = 2(1 - tanh(s))        (paper's tanh/sech parameterisation)
    p   -> phi = pi * tanh(p) ... (phase stream, here the 'up' branch scaled)
    kappa, alpha, beta are learnable per-channel parameters.
    """

    def __init__(self, dim, phase_rank=16, hidden_in=None):
        super().__init__()
        self.kappa_raw = nn.Parameter(torch.full((dim,), 2.0))        # sigmoid -> ~0.88
        self.alpha = nn.Parameter(torch.full((dim,), math.pi / 2) + 0.05 * torch.randn(dim))
        self.beta = nn.Parameter(0.05 * torch.randn(dim))
        self.phase_scale = nn.Parameter(torch.full((dim,), 1.0))
        self.phase_bias = nn.Parameter(torch.zeros(dim))

    def forward(self, s, p):
        n = 2 * (1 - torch.tanh(s))                      # (0,4)
        phi = math.pi * torch.tanh(p * self.phase_scale + self.phase_bias)
        kappa = torch.sigmoid(self.kappa_raw)
        return bloch_filter(n, phi, kappa, self.alpha, self.beta)


class APQBGatedMLP(nn.Module):
    """Wraps a Qwen-style MLP: down(act(gate(x)) * up(x)).

    out = down( up(x) * [ (1-lam)*silu(g) + lam * c * F(g, up) ] )
    lam = sigmoid(lam_raw) initialised ~0 so the pretrained model is preserved at step 0.
    """

    def __init__(self, mlp, lam_init=-8.0):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = mlp.gate_proj, mlp.up_proj, mlp.down_proj
        self.act_fn = getattr(mlp, "act_fn", Fn.silu)
        dim = self.gate_proj.out_features
        self.filt = FullAPQBFilter(dim)
        self.lam_raw = nn.Parameter(torch.full((dim,), lam_init))
        self.out_scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        g, u = self.gate_proj(x), self.up_proj(x)
        # normalise streams so tanh stays in a useful range (dtype-safe)
        gf, uf = g.float(), u.float()
        sg = gf / (gf.detach().std(-1, keepdim=True) + 1e-6)
        su = uf / (uf.detach().std(-1, keepdim=True) + 1e-6)
        F = self.filt(sg, su).to(g.dtype)
        lam = torch.sigmoid(self.lam_raw).to(g.dtype)
        gate = (1 - lam) * self.act_fn(g) + lam * self.out_scale.to(g.dtype) * F
        return self.down_proj(gate * u)


def patch_model(model, layers=None, lam_init=-8.0):
    """Replace every gated MLP (gate/up/down_proj) with APQBGatedMLP. Returns #patched."""
    targets = []
    for name, m in model.named_modules():
        if all(hasattr(m, a) for a in ("gate_proj", "up_proj", "down_proj")) and not isinstance(m, APQBGatedMLP):
            targets.append((name, m))
    count = 0
    for i, (name, m) in enumerate(targets):
        if layers is not None and i not in layers:
            continue
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, APQBGatedMLP(m, lam_init))
        count += 1
    return count


def apqb_parameters(model):
    return [p for n, p in model.named_parameters()
            if any(k in n for k in ("filt.", "lam_raw", "out_scale"))]


def freeze_except_apqb(model):
    keep = {id(p) for p in apqb_parameters(model)}
    for p in model.parameters():
        p.requires_grad = id(p) in keep


class _ChannelAffine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))


class AffineSiLUMLP(nn.Module):
    """Control for APQBGatedMLP: down( up * (a*silu(g) + b) ) with per-channel a, b.

    Identity at init. Trains the same kind of per-channel parameters as the
    filter, but without the Bloch filter, so any gain it shows is domain
    adaptation rather than an effect of the APQB filter.
    """

    def __init__(self, mlp):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = mlp.gate_proj, mlp.up_proj, mlp.down_proj
        self.act_fn = getattr(mlp, "act_fn", Fn.silu)
        self.filt = _ChannelAffine(self.gate_proj.out_features)

    def forward(self, x):
        g, u = self.gate_proj(x), self.up_proj(x)
        gate = self.filt.scale.to(g.dtype) * self.act_fn(g) + self.filt.bias.to(g.dtype)
        return self.down_proj(gate * u)


def patch_model_control(model):
    """Replace every gated MLP with AffineSiLUMLP. Returns #patched."""
    targets = [(n, m) for n, m in model.named_modules()
               if all(hasattr(m, a) for a in ("gate_proj", "up_proj", "down_proj"))
               and not isinstance(m, (APQBGatedMLP, AffineSiLUMLP))]
    for name, m in targets:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, AffineSiLUMLP(m))
    return len(targets)
