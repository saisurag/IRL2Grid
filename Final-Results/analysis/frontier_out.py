import csv, sys
from collections import defaultdict
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import os
def _find_results_csv():
    p = os.environ.get("RL2GRID_RESULTS")
    if p:
        return os.path.abspath(p)
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (os.path.join(here, os.pardir, "hpc", "results.csv"),
              os.path.join(here, os.pardir, "results", "v2_final", "results.csv")):
        if os.path.isfile(c):
            return os.path.abspath(c)
    raise SystemExit("results.csv not found; pass as argument or set RL2GRID_RESULTS")
OUT=sys.argv[1]
CSV=sys.argv[2] if len(sys.argv)>2 else _find_results_csv()
IDLE={'bus14':19.96,'bus36-M':23.80,'bus5':22.04}   # per-seed medians, 2026-08-11
cells=defaultdict(list)
for r in csv.DictReader(open(CSV)):
    if r.get('status')!='ok' or not r.get('leaves_budget'): continue
    if 'v1repro' in r['method']: continue
    cells[(r['envtag'],r['method'],int(r['leaves_budget']))].append(float(r['report']))
envs=[e for e in ('bus5','bus14','bus36-M') if any(k[0]==e for k in cells)]
colors={"distill":"#898781","viper":"#2a78d6","dtpo-full":"#e45756","dtpo":"#1baf7a",
        "dtpo-ws":"#eda100","daggertree":"#7c4dff"}
fig,axes=plt.subplots(1,len(envs),figsize=(5.4*len(envs),4.4),squeeze=False)
for ax,env in zip(axes[0],envs):
    for m in sorted({k[1] for k in cells if k[0]==env}):
        pts=sorted((k[2],np.median(v),np.percentile(v,25),np.percentile(v,75),len(v))
                   for k,v in cells.items() if k[0]==env and k[1]==m)
        if len(pts)<2: continue
        x,med,q1,q3,ns=zip(*pts)
        c=colors.get(m,"#0b0b0b")
        ax.plot(x,med,"o-",color=c,lw=2,ms=5,label=m)
        ax.fill_between(x,q1,q3,color=c,alpha=0.15,lw=0)
    ax.axhline(IDLE[env],ls="--",lw=1.5,color="#444")
    ax.text(0.02,IDLE[env],f" do-nothing {IDLE[env]:.1f}",transform=ax.get_yaxis_transform(),
            va="bottom",fontsize=8,color="#444")
    ax.set_xscale("log",base=2); ax.set_xticks(sorted({k[2] for k in cells if k[0]==env}))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("leaf budget"); ax.set_ylabel("honest REPORT survival %")
    ax.set_title(env); ax.grid(alpha=.25); ax.legend(loc="best",fontsize=7)
fig.suptitle("Survival vs interpretability budget (median, IQR band) — do-nothing = per-seed median")
fig.tight_layout(); fig.savefig(OUT,dpi=150); print("wrote",OUT)
