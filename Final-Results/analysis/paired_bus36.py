import csv, re, collections, statistics as st
import os
def _here():
    return os.path.dirname(os.path.abspath(__file__))
def _logs():
    return os.environ.get("RL2GRID_LOGS", os.path.join(_here(), "logs")) + os.sep
def _results_csv():
    p = os.environ.get("RL2GRID_RESULTS")
    if p:
        return os.path.abspath(p)
    for c in (os.path.join(_here(), os.pardir, "hpc", "results.csv"),
              os.path.join(_here(), os.pardir, "results", "v2_final", "results.csv")):
        if os.path.isfile(c):
            return os.path.abspath(c)
    raise SystemExit("results.csv not found; set RL2GRID_RESULTS")
LOGS=_logs()
IDLE=collections.defaultdict(dict)
for l in open(LOGS+'idle_per_seed.log'):
    m=re.search(r'env=(\S+) seed=(\d+).*?report=([\d.]+)',l)
    if m: IDLE[m.group(1)][int(m.group(2))]=float(m.group(3))
for env in ('bus36-M','bus5'):
    I=IDLE[env]
    print(f"\n===== {env} =====")
    print("idle by seed:", " ".join(f"{s}:{I[s]:.2f}" for s in sorted(I)))
    rows=[x for x in csv.DictReader(open(_results_csv()))
          if x['envtag']==env and x['leaves_budget']=='16' and x.get('status')=='ok']
    by=collections.defaultdict(dict)
    for x in rows: by[x['method']][int(x['seed'])]=float(x['report'])
    print(f"{'method':20s} {'n':>3s} {'paired D':>9s} {'signs':>7s} {'exact ties':>11s}   per-seed delta")
    out=[]
    for m,d in by.items():
        ss=[s for s in d if s in I]
        if not ss: continue
        dd=[d[s]-I[s] for s in ss]
        ties=sum(1 for v in dd if abs(v)<1e-9)
        out.append((st.median(dd), m, len(ss), sum(1 for v in dd if v>0), ties, dd))
    for pm,m,n,pos,ties,dd in sorted(out, reverse=True):
        print(f"{m:20s} {n:3d} {pm:+9.2f} {pos:4d}/{n} {ties:8d}/{n}   "+" ".join(f"{v:+.2f}" for v in dd))
