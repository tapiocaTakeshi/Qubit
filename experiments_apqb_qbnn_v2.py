#!/usr/bin/env python3
"""
Section 8 experimental protocol for the APQB/QBNN v2 draft.

Runs the paper's own falsifiable hypotheses (Sec. 6.7) on the synthetic
tasks of Sec. 8.2, against the Sec. 8.4 baselines and Sec. 8.5 ablations,
reporting Sec. 8.6 statistics (multi-seed mean +/- std) and evaluating
the Sec. 8.7 falsification conditions.

  H1  high-order signed interactions -> QBNN more parameter-efficient than MLP
  H2  q path -> better calibration under ambiguous labels
  H3  benefit comes from r^2+q^2=1, not merely from having two gates
      (control: Independent-gates, which has MORE parameters)

This script reports whatever it measures, including null results. Per
Sec. 8.7 a null result here is an informative outcome, not a failure.

Usage:  python3 experiments_apqb_qbnn_v2.py [--seeds 5] [--quick]
"""

import argparse
import math
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from apqb_qbnn_v2 import QBNNLayerV2, IndependentGateLayerV2

warnings.filterwarnings("ignore", category=RuntimeWarning)

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Sec. 8.4 baseline: plain MLP, width chosen to match QBNN parameters."""

    def __init__(self, in_dim, hidden, out_dim, depth=2):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.Tanh()]
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class QBNNNet(nn.Module):
    """Stack of QBNN (or Independent-gates) layers + linear read-out."""

    def __init__(self, in_dim, hidden, out_dim, depth=2, rank=8,
                 layer_cls=QBNNLayerV2, **layer_kwargs):
        super().__init__()
        blocks, d = [], in_dim
        for _ in range(depth):
            blocks.append(layer_cls(d, hidden, rank=rank, **layer_kwargs))
            d = hidden
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(d, out_dim)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.head(x)

    def mean_q(self):
        qs = [b.last_q.mean() for b in self.blocks if b.last_q is not None]
        return torch.stack(qs).mean() if qs else None


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def matched_mlp(in_dim, out_dim, target_params, depth=2):
    """Widen an MLP until its parameter count first meets/exceeds target
    (Sec. 8.4: 'MLP with parameter count matched')."""
    best = None
    for hidden in range(4, 2048):
        m = MLP(in_dim, hidden, out_dim, depth)
        n = count_params(m)
        if best is None or abs(n - target_params) < abs(best[1] - target_params):
            best = (hidden, n)
        if n >= target_params:
            break
    return best[0]


# ---------------------------------------------------------------------------
# Sec. 8.2 synthetic data
# ---------------------------------------------------------------------------

def make_parity(n, d, k, seed):
    """XOR/parity task: label = parity of the first k of d +/-1 inputs.
    Sec. 8.2 row 'XOR / parity' -- probes high-order signed interactions."""
    g = torch.Generator().manual_seed(seed)
    x = (torch.randint(0, 2, (n, d), generator=g).float() * 2 - 1)
    y = (x[:, :k].prod(dim=1) > 0).long()
    return x, y


def make_ambiguous(n, d, seed, noise_scale=0.9):
    """Input-dependent label noise: points near the decision boundary get
    flipped more often. Sec. 8.2 row 'ambiguous labels' -- probes whether
    the q path captures uncertainty (H2)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    margin = x[:, 0]
    y_clean = (margin > 0).long()
    # flip prob is highest at the boundary (|margin| ~ 0)
    p_flip = noise_scale * torch.exp(-4.0 * margin ** 2) * 0.5
    flip = torch.rand(n, generator=g) < p_flip
    y = torch.where(flip, 1 - y_clean, y_clean)
    return x, y


# ---------------------------------------------------------------------------
# Sec. 8.6 metrics
# ---------------------------------------------------------------------------

def expected_calibration_error(probs, labels, n_bins=15):
    conf, pred = probs.max(dim=1)
    correct = (pred == labels).float()
    ece = torch.zeros(())
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.float().mean() * (correct[m].mean() - conf[m].mean()).abs()
    return ece.item()


def brier_score(probs, labels):
    onehot = F.one_hot(labels, probs.shape[1]).float()
    return ((probs - onehot) ** 2).sum(dim=1).mean().item()


def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        nll = F.cross_entropy(logits, y).item()
        acc = (logits.argmax(1) == y).float().mean().item()
    return {
        "acc": acc,
        "nll": nll,
        "ece": expected_calibration_error(probs, y),
        "brier": brier_score(probs, y),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, xtr, ytr, xva, yva, xte, yte, epochs=200, bs=256, lr=3e-3,
          wd=1e-5, eval_every=5):
    """Trains with validation-based model selection (best val NLL).

    Without this, a comparison under label noise degenerates into
    'which model overfits fastest' rather than which generalizes -- the
    multiplicative gate of Sec. 6.2 is explicitly flagged in Sec. 10 as
    prone to amplification, so every model gets the same early-stopping
    protection to keep the comparison about generalization.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    n = xtr.shape[0]
    best_nll, best_state = float("inf"), None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(model(xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        if ep % eval_every == 0 or ep == epochs - 1:
            v = evaluate(model, xva, yva)
            if v["nll"] < best_nll:
                best_nll = v["nll"]
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return evaluate(model, xte, yte)


def agg(runs, key):
    v = torch.tensor([r[key] for r in runs])
    return v.mean().item(), v.std(unbiased=len(v) > 1).item()


def fmt(runs, key, pct=False):
    m, s = agg(runs, key)
    return f"{m*100:5.1f} +/- {s*100:4.1f}" if pct else f"{m:.4f} +/- {s:.4f}"


# ---------------------------------------------------------------------------
# Experiment drivers
# ---------------------------------------------------------------------------

def build_models(in_dim, out_dim, hidden, depth, rank):
    """Sec. 8.5 ablation table + Sec. 8.4 baselines."""
    gate_kw = dict(lambda_r_init=0.01, lambda_q_init=0.01)
    specs = {
        "QBNN-full":   lambda: QBNNNet(in_dim, hidden, out_dim, depth, rank, **gate_kw),
        "QBNN-r":      lambda: QBNNNet(in_dim, hidden, out_dim, depth, rank, use_q=False, **gate_kw),
        "QBNN-q":      lambda: QBNNNet(in_dim, hidden, out_dim, depth, rank, use_r=False, **gate_kw),
        "Indep-gates": lambda: QBNNNet(in_dim, hidden, out_dim, depth, rank,
                                       layer_cls=IndependentGateLayerV2, **gate_kw),
        "Random-J":    lambda: QBNNNet(in_dim, hidden, out_dim, depth, rank,
                                       learnable_J=False, **gate_kw),
    }
    ref = specs["QBNN-full"]()
    target = count_params(ref)
    mlp_hidden = matched_mlp(in_dim, out_dim, target, depth)
    specs["MLP-matched"] = lambda: MLP(in_dim, mlp_hidden, out_dim, depth)
    specs["MLP-same-width"] = lambda: MLP(in_dim, hidden, out_dim, depth)
    return specs, target, mlp_hidden


def run_task(name, make_data, in_dim, seeds, hidden, depth, rank, epochs,
             n_train, n_test, models_subset=None):
    print(f"\n{'='*72}\n{name}\n{'='*72}")
    specs, target, mlp_hidden = build_models(in_dim, 2, hidden, depth, rank)
    if models_subset:
        specs = {k: v for k, v in specs.items() if k in models_subset}

    print(f"QBNN-full params: {target}   MLP-matched hidden width: {mlp_hidden}"
          f"   (QBNN hidden {hidden}, depth {depth}, rank {rank})")

    results = {}
    for mname, ctor in specs.items():
        runs, t0 = [], time.time()
        for s in range(seeds):
            torch.manual_seed(1000 + s)
            xtr, ytr = make_data(n_train, s)
            xva, yva = make_data(max(500, n_train // 4), 250 + s)
            xte, yte = make_data(n_test, 500 + s)
            model = ctor().to(DEVICE)
            runs.append(train(model, xtr, ytr, xva, yva, xte, yte, epochs=epochs))
        results[mname] = runs
        p = count_params(ctor())
        print(f"  {mname:<16} params={p:<7} acc={fmt(runs,'acc',True)}  "
              f"nll={fmt(runs,'nll')}  ece={fmt(runs,'ece')}  "
              f"brier={fmt(runs,'brier')}  [{time.time()-t0:.0f}s]")
    return results


def run_sample_efficiency(name, make_data, in_dim, sizes, seeds, hidden, depth,
                          rank, epochs, n_test, models_subset):
    """Sec. 8.2 lists 'sample efficiency' (not single-point accuracy) as the
    metric for XOR/parity. Parity has a sharp all-or-nothing transition, so
    a single training-set size either saturates every model at 100% or
    floors every model at chance; sweeping n_train is the only way to read
    off a real difference, and it avoids picking the one size that happens
    to flatter a given model."""
    print(f"\n{'='*72}\n{name}\n{'='*72}")
    specs, target, mlp_hidden = build_models(in_dim, 2, hidden, depth, rank)
    specs = {k: v for k, v in specs.items() if k in models_subset}
    print(f"QBNN-full params: {target}   MLP-matched hidden width: {mlp_hidden}")
    print(f"\n  {'n_train':<9}" + "".join(f"{m:>18}" for m in specs))

    table = {m: {} for m in specs}
    for n_train in sizes:
        row = []
        for mname, ctor in specs.items():
            accs = []
            for s in range(seeds):
                torch.manual_seed(1000 + s)
                xtr, ytr = make_data(n_train, s)
                xva, yva = make_data(max(500, n_train // 4), 250 + s)
                xte, yte = make_data(n_test, 500 + s)
                res = train(ctor().to(DEVICE), xtr, ytr, xva, yva, xte, yte,
                            epochs=epochs)
                accs.append(res["acc"])
            a = torch.tensor(accs)
            table[mname][n_train] = (a.mean().item(), a.std(unbiased=len(a) > 1).item())
            row.append(f"{a.mean()*100:11.1f} +/-{a.std(unbiased=len(a)>1)*100:4.1f}")
        print(f"  {n_train:<9}" + "".join(f"{c:>18}" for c in row))
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    seeds = 3 if args.quick else args.seeds
    epochs = 80 if args.quick else 200

    print("APQB/QBNN v2 -- Section 8 experimental protocol")
    print(f"seeds={seeds}  epochs={epochs}  device={DEVICE}")

    # --- H1: high-order signed interactions (parity) ----------------------
    d, k = 16, 5
    sizes = [500, 1000, 1500, 2000] if args.quick else [500, 750, 1000, 1500, 2000, 3000]
    parity = run_sample_efficiency(
        f"Task A (H1): sample efficiency on parity of first {k} of {d} inputs"
        f"  [Sec. 8.2 'XOR/parity']",
        lambda n, s: make_parity(n, d, k, s),
        in_dim=d, sizes=sizes, seeds=seeds, hidden=48, depth=2, rank=8,
        epochs=epochs, n_test=4000,
        models_subset=["QBNN-full", "QBNN-r", "Indep-gates", "MLP-matched"])

    # --- H2: ambiguous labels / calibration -------------------------------
    d2 = 8
    ambig = run_task(
        "Task B (H2): input-dependent label noise  [Sec. 8.2 'ambiguous labels']",
        lambda n, s: make_ambiguous(n, d2, s),
        in_dim=d2, seeds=seeds, hidden=48, depth=2, rank=8,
        epochs=epochs, n_train=3000, n_test=3000)

    # --- Sec. 8.7 falsification assessment --------------------------------
    print(f"\n{'='*72}\nSec. 8.7 falsification conditions\n{'='*72}")

    def m(res, name, key):
        return agg(res[name], key)[0]

    # H1: sample efficiency of QBNN vs parameter-matched MLP on parity.
    # Compare over the whole sweep, and only count sizes where the task is
    # actually informative (not pinned at ceiling or chance for both).
    print("\nH1  parity sample-efficiency sweep (QBNN-full vs MLP-matched):")
    wins = losses = ties = informative = 0
    for n_train in sorted(parity["QBNN-full"]):
        q = parity["QBNN-full"][n_train][0]
        b = parity["MLP-matched"][n_train][0]
        if min(q, b) > 0.99 or max(q, b) < 0.60:
            tag = "uninformative (ceiling/floor)"
        else:
            informative += 1
            if q > b + 0.02:
                tag, _ = "QBNN ahead", (wins := wins + 1)
            elif b > q + 0.02:
                tag, _ = "MLP ahead", (losses := losses + 1)
            else:
                tag, _ = "tie", (ties := ties + 1)
        print(f"    n={n_train:<6} QBNN {q*100:5.1f}%   MLP {b*100:5.1f}%   -> {tag}")
    if informative == 0:
        print("    -> UNINFORMATIVE: no training-set size separates the models.")
    elif wins > losses:
        print(f"    -> H1 SUPPORTED ({wins} wins / {losses} losses / {ties} ties)")
    elif losses >= wins and losses > 0:
        print(f"    -> H1 FALSIFIED on this task ({losses} losses / {wins} wins): "
              "matched MLP is at least as sample-efficient (Sec. 8.7 condition 1)")
    else:
        print(f"    -> H1 INCONCLUSIVE ({ties} ties, no size shows a >2-point gap)")

    # H2: q path and calibration under ambiguous labels
    full_ece = m(ambig, "QBNN-full", "ece")
    r_only_ece = m(ambig, "QBNN-r", "ece")
    mlp_ece = m(ambig, "MLP-matched", "ece")
    print(f"\nH2  ambiguous-label ECE: QBNN-full {full_ece:.4f}  "
          f"QBNN-r (no q) {r_only_ece:.4f}  MLP-matched {mlp_ece:.4f}")
    if full_ece < r_only_ece and full_ece < mlp_ece:
        print("    -> H2 SUPPORTED on this task (q path improves calibration)")
    else:
        print("    -> H2 NOT SUPPORTED on this task: q path did not reduce ECE "
              "(Sec. 8.7 condition 3)")

    # H3: constraint vs independent gates
    print("\nH3  APQB constraint (r^2+q^2=1) vs Independent-gates control:")
    for n_train in sorted(parity["QBNN-full"]):
        f_acc = parity["QBNN-full"][n_train][0]
        i_acc = parity["Indep-gates"][n_train][0]
        verdict = ("constraint ahead" if f_acc > i_acc + 0.02 else
                   "control ahead" if i_acc > f_acc + 0.02 else "tie")
        print(f"    parity n={n_train:<6} QBNN-full {f_acc*100:5.1f}%  "
              f"Indep-gates {i_acc*100:5.1f}%  -> {verdict}")
    f_acc = m(ambig, "QBNN-full", "acc")
    i_acc = m(ambig, "Indep-gates", "acc")
    verdict = ("constraint ahead" if f_acc > i_acc + 0.01 else
               "control ahead" if i_acc > f_acc + 0.01 else "tie")
    print(f"    ambiguous     QBNN-full {f_acc*100:5.1f}%  "
          f"Indep-gates {i_acc*100:5.1f}%  -> {verdict}")
    print("    (Sec. 8.7 condition 2: if Indep-gates is consistently ahead, "
          "the APQB constraint is not what helps)")

    print("\nNote: these are small synthetic probes, not a validation of the "
          "architecture. Sec. 8.3's real-data protocol (UCI/OpenML, multiple "
          "datasets, mean ranks) is still required before any claim.")


if __name__ == "__main__":
    main()
