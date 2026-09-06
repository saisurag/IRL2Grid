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
IDLE={100:17.02}
for f in ('idle_per_seed.log',):
    for l in open(LOGS+f):
        m=re.search(r'env=bus14 seed=(\d+).*?report=([\d.]+)',l)
        if m: IDLE[int(m.group(1))]=float(m.group(2))
r=[IDLE[s] for s in sorted(IDLE)]
print("bus14 IDLE by seed 100-109:", " ".join(f"{v:.2f}" for v in r))
print(f"  median {st.median(r):.2f}  min {min(r):.2f}  max {max(r):.2f}  spread {max(r)-min(r):.2f}\n")
rows=[x for x in csv.DictReader(open(_results_csv()))
      if x['envtag']=='bus14' and x['leaves_budget']=='16' and x.get('status')=='ok'
      and 'v1repro' not in x['method']]
by=collections.defaultdict(dict)
for x in rows: by[x['method']][int(x['seed'])]=float(x['report'])
# dagger has no leaves budget
for x in csv.DictReader(open(_results_csv())):
    if x['envtag']=='bus14' and x['method']=='dagger' and x.get('status')=='ok':
        by['dagger'][int(x['seed'])]=float(x['report'])
print(f"{'method':20s} {'n':>3s} {'median':>7s} {'naive D vs 17.02':>16s} {'PAIRED D vs idle':>16s} {'signs':>7s}")
out=[]
for m,d in by.items():
    ss=[s for s in d if s in IDLE]
    if not ss: continue
    med=st.median(d[s] for s in ss)
    pd=[d[s]-IDLE[s] for s in ss]
    out.append((st.median(pd), m, len(ss), med, med-17.02, sum(1 for v in pd if v>0)))
for pm,m,n,med,naive,pos in sorted(out, reverse=True):
    print(f"{m:20s} {n:3d} {med:7.2f} {naive:+16.2f} {pm:+16.2f} {pos:4d}/{n}")
