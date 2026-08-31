#!/usr/bin/env python3

import sys, glob, os, json, warnings, numpy as np, torch as th
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env.config, env.utils
from env.eval import Evaluator
from alg.dtpo.agent import RegressionTreePolicy

GAMMA = 0.99
RUNS = [f"hpc/runs/bus14_dtpo_L16_s{s}" for s in range(100, 110)]

sys.argv = ["x","--env-id","bus14","--difficulty","0","--action-type","topology",
            "--chronic-holdout","4","--eval-seed","12345","--eval-norm-reset","per-chronic",
            "--chronics-mode","multifolder","--use-heuristic","False","--heuristic-type","reconnect"]
a = env.config.get_env_args(); a.seed, a.alg, a.cuda, a.track = 100, "DTPO", False, False
ev = Evaluator(a, None, th.device("cpu"))
ids = ev.fixed_chronic_ids(80)
sel_ids, rep_ids = ids[0::2], ids[1::2]
print(f"[setup] select={len(sel_ids)} report={len(rep_ids)} chronics, gamma={GAMMA}", flush=True)

rows = []
for rd in RUNS:
    cks = glob.glob(rd + "/checkpoint/*.tar")
    if not cks:
        print(f"  {rd}: no checkpoint", flush=True); continue
    fin = [c for c in cks if os.path.basename(c).startswith("final_")]
    rec = th.load(max(fin or cks, key=os.path.getmtime), map_location="cpu", weights_only=False)
    cands = rec.get("candidates") or []
    if not cands:
        print(f"  {rd}: no candidates", flush=True); continue

    surv, dret = [], []
    for c in cands:
        pol = RegressionTreePolicy(c["tree"].n_outputs_, th.device("cpu"),
                                   tree=c["tree"], action_map=rec.get("action_kept"))
        r = ev.evaluate_fixed(pol, sel_ids, disc_gamma=GAMMA)
        surv.append(r["survival"]); dret.append(r["disc_return"])

    i_s, i_r = int(np.argmax(surv)), int(np.argmax(dret))
    def report(i):
        pol = RegressionTreePolicy(cands[i]["tree"].n_outputs_, th.device("cpu"),
                                   tree=cands[i]["tree"], action_map=rec.get("action_kept"))
        return 100.0 * ev.evaluate_fixed(pol, rep_ids)["survival"]
    rep_s = report(i_s)
    rep_r = rep_s if i_r == i_s else report(i_r)

    seed = int(rd.split("_s")[-1])
    rows.append(dict(seed=seed, idx_surv=i_s, idx_ret=i_r, same=(i_s == i_r),
                     report_surv=rep_s, report_ret=rep_r,
                     sel_surv=[100*x for x in surv], sel_dret=dret))
    print(f"  seed {seed}: survival picks cand {i_s}, return picks cand {i_r}"
          f"{'  (same)' if i_s==i_r else '  <-- DIFFERENT'}"
          f"   report {rep_s:.2f} vs {rep_r:.2f}", flush=True)

json.dump(rows, open("select_by_return_results.json","w"), indent=1)
if rows:
    d = [r["report_ret"] - r["report_surv"] for r in rows]
    import statistics as st
    print(f"\n=== {len(rows)} seeds ===")
    print(f"  criterion changed the chosen tree in {sum(1 for r in rows if not r['same'])}/{len(rows)} runs")
    print(f"  report survival, survival-selected: median {st.median(r['report_surv'] for r in rows):.2f}")
    print(f"  report survival, return-selected  : median {st.median(r['report_ret'] for r in rows):.2f}")
    print(f"  paired (return - survival): median {st.median(d):+.2f}  mean {st.mean(d):+.2f}  "
          f"positive {sum(1 for x in d if x>0)}/{len(d)}")
