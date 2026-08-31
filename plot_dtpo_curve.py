#!/usr/bin/env python
"""
Parse a DTPO training log and plot survival% (per-eval and running-best) vs iteration. Proof-of-training artifact for each run.
Example: python plot_dtpo_curve.py --log run/train.log --out run/curve.png
"""
import argparse, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--log", required=True)
p.add_argument("--out", required=True)
p.add_argument("--title", default=None)
args = p.parse_args()

# [DTPO] it=15/80 step=30000 J_new=0.1748 kept leaves=16 survival=100.00% best=100.00%
pat = re.compile(r"\[DTPO\]\s+it=(\d+)/(\d+).*?survival=([\d.]+)%\s+best=([\d.]+)%")
its, surv, best, leaves, total = [], [], [], None, None
leaf_pat = re.compile(r"kept leaves=(\d+)")
with open(args.log) as fh:
    for ln in fh:
        m = pat.search(ln)
        if m:
            its.append(int(m.group(1))); total = int(m.group(2))
            surv.append(float(m.group(3))); best.append(float(m.group(4)))
            lm = leaf_pat.search(ln)
            if lm: leaves = int(lm.group(1))

if not its:
    raise SystemExit(f"No '[DTPO] it=' lines found in {args.log} — did the run print evals?")

plt.figure(figsize=(7, 4.2))
plt.plot(its, surv, "o-", color="#4c78a8", lw=1.6, ms=4, label="eval survival")
plt.plot(its, best, "-", color="#e45756", lw=2.2, label="running best")
plt.axhline(100, color="#999", ls=":", lw=1)
plt.ylim(-3, 105)
plt.xlabel("DTPO iteration"); plt.ylabel("survival %")
ttl = args.title or "DTPO training"
if leaves is not None: ttl += f"  ({leaves} leaves)"
plt.title(ttl)
plt.legend(loc="lower right"); plt.grid(alpha=0.25)
plt.tight_layout(); plt.savefig(args.out, dpi=140)
print(f"[curve] {args.out}  final_best={best[-1]:.1f}%" f"first_100_at_it={next((i for i,b in zip(its,best) if b>=100), None)}  leaves={leaves}")
