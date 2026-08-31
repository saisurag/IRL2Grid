#!/usr/bin/env python3

import argparse
import csv
import os
import re
from collections import defaultdict

import numpy as np


def parse_kv(line):
    return dict(kv.split("=", 1) for kv in line.strip().split()[1:] if "=" in kv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=os.path.join(os.path.dirname(__file__), "runs"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results.csv"))
    a = ap.parse_args()

    rows = []
    for name in sorted(os.listdir(a.runs_dir)) if os.path.isdir(a.runs_dir) else []:
        d = os.path.join(a.runs_dir, name)
        helog = os.path.join(d, "honest_eval.log")
        if not os.path.isfile(helog):
            rows.append({"name": name, "status": "no_eval"})
            continue
        result = None
        with open(helog) as f:
            for line in f:
                if line.startswith("RESULT "):
                    result = parse_kv(line)
        if result is None:
            rows.append({"name": name, "status": "eval_failed"})
            continue

        m = re.match(r"^(?P<envtag>[^_]+)_(?P<method>.+?)(?:_L(?P<leaves>\d+))?_s(?P<seed>\d+)$", name)
        meta = m.groupdict() if m else {"envtag": "?", "method": "?", "leaves": None, "seed": "?"}

        jumps = {}
        jpath = os.path.join(d, "jumps.txt")
        if os.path.isfile(jpath):
            with open(jpath) as f:
                jumps = parse_kv(f.readline().replace("JUMPS", "JUMPS "))

        rows.append({
            "name": name, "status": "ok",
            "envtag": meta["envtag"], "method": meta["method"],
            "leaves_budget": meta["leaves"], "seed": meta["seed"],
            "alg": result.get("alg"), "env": result.get("env"),
            "difficulty": result.get("difficulty"),
            "leaves_actual": result.get("leaves"), "depth": result.get("depth"),
            "select": result.get("select"), "report": result.get("report"),
            "report_min": result.get("report_min"), "report_max": result.get("report_max"),
            "frac100": result.get("frac100"), "n_report": result.get("n_report"),
            "bigjumps": jumps.get("bigjumps"), "candjumps": jumps.get("candjumps"),
        })

    if not rows:
        print(f"nothing found under {a.runs_dir}")
        return

    fields = ["name", "status", "envtag", "method", "leaves_budget", "seed", "alg", "env",
              "difficulty", "leaves_actual", "depth", "select", "report", "report_min",
              "report_max", "frac100", "n_report", "bigjumps", "candjumps"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    ok = [r for r in rows if r["status"] == "ok"]
    print(f"wrote {a.out}: {len(ok)} ok / {len(rows)} total\n")

    cells = defaultdict(list)
    for r in ok:
        cells[(r["envtag"], r["method"], r["leaves_budget"] or "-")].append(r)
    print(f"{'env':<10} {'method':<20} {'leaves':<7} {'n':<3} "
          f"{'report median [IQR]':<24} {'min..max':<15} {'bigjumps':<8}")
    print("-" * 90)
    for (env, method, leaves), rs in sorted(cells.items()):
        rep = np.array([float(r["report"]) for r in rs])
        q1, med, q3 = np.percentile(rep, [25, 50, 75])
        bj = [float(r["bigjumps"]) for r in rs if r.get("bigjumps") not in (None, "")]
        bjs = f"{np.mean(bj):.1f}" if bj else "-"
        print(f"{env:<10} {method:<20} {leaves:<7} {len(rs):<3} "
              f"{med:6.2f} [{q1:6.2f},{q3:6.2f}]   {rep.min():6.2f}..{rep.max():6.2f}  {bjs:<8}")

    bad = [r for r in rows if r["status"] != "ok"]
    if bad:
        print(f"\nincomplete ({len(bad)}): " + ", ".join(r["name"] for r in bad[:20])
              + (" ..." if len(bad) > 20 else ""))


if __name__ == "__main__":
    main()
