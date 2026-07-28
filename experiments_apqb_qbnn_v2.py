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
    to flatter a given model.

    Per size and model we record (mean acc, std, solve rate). Because each
    seed either finds the parity circuit or sits at chance, the mean is a
    mixture of two modes and its std is large by construction -- the solve
    rate (fraction of seeds above 90%) is the honest summary statistic.
    """
    print(f"\n{'='*72}\n{name}\n{'='*72}")
    specs, target, mlp_hidden = build_models(in_dim, 2, hidden, depth, rank)
    specs = {k: v for k, v in specs.items() if k in models_subset}
    print(f"QBNN-full params: {target}   MLP-matched hidden width: {mlp_hidden}")
    for mname, ctor in specs.items():
        print(f"    {mname:<16} params={count_params(ctor())}")
    print(f"\n  {'n_train':<9}" + "".join(f"{m:>22}" for m in specs))

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
            sd = a.std(unbiased=len(a) > 1).item()
            solved = (a > 0.9).float().mean().item()
            table[mname][n_train] = (a.mean().item(), sd, solved)
            row.append(f"{a.mean()*100:5.1f}+/-{sd*100:4.1f} [{solved*100:3.0f}%]")
        print(f"  {n_train:<9}" + "".join(f"{c:>22}" for c in row))
    print("  (cell = mean acc +/- std  [solve rate: seeds reaching >90% acc])")
    return table


# ---------------------------------------------------------------------------
# Sec. 8.7 falsification assessment
# ---------------------------------------------------------------------------

SOLVE_MARGIN = 0.15   # solve-rate gap counted as a real difference


def assess_h1(parity):
    """Sec. 8.7 condition 1: is QBNN more sample-efficient than a
    parameter-matched MLP on high-order signed interactions?"""
    print("\nH1  parity sample efficiency, by solve rate "
          "(fraction of seeds reaching >90% acc):")
    wins = losses = ties = informative = narrow_losses = 0
    for n in sorted(parity["QBNN-full"]):
        q = parity["QBNN-full"][n][2]
        b = parity["MLP-matched"][n][2]
        w = parity["MLP-same-width"][n][2]
        if min(q, b, w) > 0.99 or max(q, b, w) < 0.01:
            tag = "uninformative"
        else:
            informative += 1
            if q > b + SOLVE_MARGIN:
                tag = "QBNN ahead"
                wins += 1
            elif b > q + SOLVE_MARGIN:
                tag = "MLP ahead"
                losses += 1
            else:
                tag = "tie"
                ties += 1
            if w > q + SOLVE_MARGIN:
                narrow_losses += 1
        print(f"    n={n:<6} QBNN {q*100:3.0f}%   MLP-matched {b*100:3.0f}%   "
              f"MLP-same-width {w*100:3.0f}%   -> {tag}")

    if informative == 0:
        print("    -> UNINFORMATIVE: no training-set size separates the models.")
    elif wins > losses:
        print(f"    -> H1 SUPPORTED ({wins} wins / {losses} losses / {ties} ties)")
    elif losses > 0:
        print(f"    -> H1 FALSIFIED on this task ({losses} losses / {wins} wins): "
              "the matched MLP is at least as sample-efficient "
              "(Sec. 8.7 condition 1)")
    else:
        print(f"    -> H1 INCONCLUSIVE ({ties} ties)")

    if narrow_losses:
        print(f"    -> STRONGER: at {narrow_losses} size(s) an MLP of the SAME "
              "WIDTH (far fewer parameters) also beat QBNN, so the loss is not "
              "an artifact of parameter-matching handing the MLP more width.")
    else:
        print("    -> CAVEAT: MLP-same-width did not beat QBNN, so part of any "
              "MLP-matched advantage may be a width effect rather than the gate.")


def assess_h2(ambig):
    """Sec. 8.7 condition 3: does the q path improve calibration?"""
    def mean_of(name, key):
        return agg(ambig[name], key)[0]

    full_ece = mean_of("QBNN-full", "ece")
    r_only_ece = mean_of("QBNN-r", "ece")
    mlp_ece = mean_of("MLP-matched", "ece")
    accs = [mean_of(n, "acc") for n in ambig]
    spread = max(accs) - min(accs)

    print(f"\nH2  ambiguous-label ECE: QBNN-full {full_ece:.4f}  "
          f"QBNN-r (no q) {r_only_ece:.4f}  MLP-matched {mlp_ece:.4f}")
    if spread < 0.01:
        print(f"    -> UNINFORMATIVE: every model lands within {spread*100:.1f} "
              "accuracy points, i.e. all of them sit at this task's Bayes "
              "limit, so the task cannot resolve a q-path effect either way.")
        return
    if full_ece < r_only_ece and full_ece < mlp_ece:
        print("    -> H2 SUPPORTED on this task (q path improves calibration)")
    else:
        print("    -> H2 NOT SUPPORTED on this task: q path did not reduce ECE "
              "(Sec. 8.7 condition 3)")


def assess_h3(parity, ambig):
    """Sec. 8.7 condition 2: is it the r^2+q^2=1 constraint that helps, or
    merely having two multiplicative gates?"""
    print("\nH3  APQB constraint (r^2+q^2=1) vs Independent-gates control:")
    ahead = behind = 0
    if parity:
        for n in sorted(parity["QBNN-full"]):
            f = parity["QBNN-full"][n][2]
            i = parity["Indep-gates"][n][2]
            if f > i + SOLVE_MARGIN:
                verdict = "constraint ahead"
                ahead += 1
            elif i > f + SOLVE_MARGIN:
                verdict = "control ahead"
                behind += 1
            else:
                verdict = "tie"
            print(f"    parity n={n:<6} solve rate: QBNN-full {f*100:3.0f}%  "
                  f"Indep-gates {i*100:3.0f}%  -> {verdict}")
    if ambig:
        f = agg(ambig["QBNN-full"], "acc")[0]
        i = agg(ambig["Indep-gates"], "acc")[0]
        verdict = ("constraint ahead" if f > i + 0.01 else
                   "control ahead" if i > f + 0.01 else "tie")
        print(f"    ambiguous      acc:        QBNN-full {f*100:5.1f}%  "
              f"Indep-gates {i*100:5.1f}%  -> {verdict}")

    if ahead == behind == 0:
        print("    -> H3 UNSUPPORTED: constraint and control are "
              "indistinguishable everywhere; nothing is attributable to "
              "r^2+q^2=1 on these tasks.")
    elif behind >= ahead:
        print(f"    -> H3 FALSIFIED ({behind} control wins / {ahead} constraint "
              "wins): removing the constraint does not hurt (Sec. 8.7 cond. 2)")
    else:
        print(f"    -> H3 WEAKLY SUPPORTED ({ahead} constraint wins / {behind} "
              "control wins) -- but see the per-size spread before believing it.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--task", choices=["a", "b", "both"], default="both",
                    help="run only the parity sweep (a), only the "
                         "ambiguous-label task (b), or both")
    args = ap.parse_args()

    seeds = 3 if args.quick else args.seeds
    epochs = 80 if args.quick else 200

    print("APQB/QBNN v2 -- Section 8 experimental protocol")
    print(f"seeds={seeds}  epochs={epochs}  task={args.task}  device={DEVICE}")

    # --- H1: high-order signed interactions (parity) ----------------------
    d, k = 16, 5
    sizes = ([500, 1000, 1500, 2000] if args.quick
             else [500, 750, 1000, 1500, 2000, 3000])
    parity = None if args.task == "b" else run_sample_efficiency(
        f"Task A (H1): sample efficiency on parity of first {k} of {d} inputs"
        f"  [Sec. 8.2 'XOR/parity']",
        lambda n, s: make_parity(n, d, k, s),
        in_dim=d, sizes=sizes, seeds=seeds, hidden=48, depth=2, rank=8,
        epochs=epochs, n_test=4000,
        # MLP-same-width is the decisive control: parameter-matching hands
        # the MLP nearly double the width (88 vs 48) and parity is strongly
        # width-sensitive, so without it a QBNN loss cannot be attributed to
        # the correlation gate rather than to the width gap.
        models_subset=["QBNN-full", "QBNN-r", "Indep-gates",
                       "MLP-matched", "MLP-same-width"])

    # --- H2: ambiguous labels / calibration -------------------------------
    d2 = 8
    ambig = None if args.task == "a" else run_task(
        "Task B (H2): input-dependent label noise  [Sec. 8.2 'ambiguous labels']",
        lambda n, s: make_ambiguous(n, d2, s),
        in_dim=d2, seeds=seeds, hidden=48, depth=2, rank=8,
        epochs=epochs, n_train=3000, n_test=3000)

    # --- Sec. 8.7 falsification assessment --------------------------------
    print(f"\n{'='*72}\nSec. 8.7 falsification conditions\n{'='*72}")
    if parity:
        assess_h1(parity)
    if ambig:
        assess_h2(ambig)
    if parity or ambig:
        assess_h3(parity, ambig)

    print("\nNote: these are small synthetic probes, not a validation of the "
          "architecture. Sec. 8.3's real-data protocol (UCI/OpenML, multiple "
          "datasets, mean ranks) is still required before any claim.")


if __name__ == "__main__":
    main()
