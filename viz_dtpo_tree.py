#!/usr/bin/env python
"""Export a trained DTPO tree to a clean Graphviz .dot / .svg.
  --> decision node : "obs_i <= thr"
  --> leaf          : "-> a=<argmax action>  p=<prob mass>"
Leaves are coloured by their chosen action so repeated actions are easy to spot.
"""
import argparse, os, shutil, subprocess, colorsys
import numpy as np
import torch as th

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--out", default=None, help="output basename (default: derived from ckpt)")
p.add_argument("--max-depth", type=int, default=None, help="truncate drawing below this depth")
p.add_argument("--landscape", action="store_true", help="left-to-right layout (wider, less tall)")
p.add_argument("--feature-names", default=None, help="optional comma-separated obs names (else obs_0..obs_n)")
args = p.parse_args()

rec = th.load(args.ckpt, map_location="cpu", weights_only=False)
tree = rec.get("tree")
if tree is None:
    raise SystemExit(f"No 'tree' in {args.ckpt} (keys {list(rec.keys())}) — not a DTPO checkpoint?")
amap = rec.get("action_map")
amap = None if amap is None else np.asarray(amap)

t = tree.tree_
n_feat = int(tree.n_features_in_)
fnames = (args.feature_names.split(",") if args.feature_names
          else [f"obs_{i}" for i in range(n_feat)])

# stable distinct colour per chosen action
def colour(a):
    h = (a * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.28, 1.0)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

lines = ['digraph dtpo {',
         f'  rankdir={"LR" if args.landscape else "TB"};',
         '  graph [fontname="Helvetica", nodesep=0.25, ranksep=0.45, bgcolor="white"];',
         '  node  [fontname="Helvetica", fontsize=11, penwidth=1.0];',
         '  edge  [fontname="Helvetica", fontsize=10, color="#555555"];']

def is_leaf(n): return t.children_left[n] == t.children_right[n]

def emit(n, depth):
    if is_leaf(n) or (args.max_depth is not None and depth >= args.max_depth):
        val = t.value[n].reshape(-1)
        a = int(np.argmax(val))
        env_a = int(amap[a]) if amap is not None else a
        prob = val[a] / (val.sum() + 1e-12)
        lbl = f"action {env_a}\\np={prob:.2f}"
        lines.append(f'  {n} [label="{lbl}", shape=box, style="filled,rounded", 'f'fillcolor="{colour(env_a)}"];')
        return
    f = fnames[t.feature[n]]
    thr = t.threshold[n]
    lines.append(f'  {n} [label="{f} \\u2264 {thr:.3g}", shape=oval, 'f'style=filled, fillcolor="#eef3fb"];')
    l, r = t.children_left[n], t.children_right[n]
    lines.append(f'  {n} -> {l} [label="yes"];')
    lines.append(f'  {n} -> {r} [label="no"];')
    emit(l, depth + 1); emit(r, depth + 1)

emit(0, 0)
lines.append('}')

base = args.out or os.path.splitext(os.path.basename(args.ckpt))[0] + "_tree"
dot_path = base + ".dot"
with open(dot_path, "w") as fh:
    fh.write("\n".join(lines))
print(f"[viz] {dot_path}  (depth={tree.get_depth()} leaves={tree.get_n_leaves()} "f"actions={tree.n_outputs_}{' landscape' if args.landscape else ''})")

dot_bin = shutil.which("dot")
if dot_bin:
    svg_path = base + ".svg"
    subprocess.run([dot_bin, "-Tsvg", dot_path, "-o", svg_path], check=True)
    print(f"[viz] {svg_path}")
else:
    print(f"[viz] `dot` not found — run: dot -Tsvg {dot_path} -o {base}.svg")
