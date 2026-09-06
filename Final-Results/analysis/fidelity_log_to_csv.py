#!/usr/bin/env python3
"""Convert a fidelity_winner.py log into CSV, one row per run.

Usage: fidelity_log_to_csv.py <log> <out.csv>
"""
import sys, re, csv

FIELDS = ["run","arm","env","seed","picked","leaves","depth","episodes","collect_seed",
          "orac_fid","orac_wfid","orac_n","orac_surv",
          "tree_fid","tree_wfid","tree_n","tree_surv","secs"]

def parse(line):
    if not line.startswith("FIDELITY "):
        return None
    kv = {}
    for tok in line.split()[1:]:
        if "=" in tok:
            k, v = tok.split("=", 1)
            kv[k] = v
    run = kv.get("run", "")
    m = re.match(r"(?P<env>bus[\w.-]*?)_(?P<arm>.+)_L(?P<lb>\d+)_s(?P<sd>\d+)$", run)
    if m:
        kv["env"], kv["arm"] = m.group("env"), m.group("arm")
    return {f: kv.get(f, "") for f in FIELDS}

def main(src, dst):
    rows = [r for r in (parse(l) for l in open(src)) if r]
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} rows -> {dst}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
