#!/usr/bin/env python3
"""Markdown tables in the BEFORE.md format from run_matrix.sh output: per mode, np x threads,
median [min-max] of the repeats, "(=)" when every repeat agrees to the printed 4 digits.
Usage: matrix_table.py matrix.txt [modes...]"""
import re, sys, statistics
from collections import defaultdict
cols = ["dP", "dX", "dXpos", "dL", "dLcm", "dLvel"]
rows = defaultdict(list)
errors = defaultdict(int)
for line in open(sys.argv[1]):
    if not line.startswith("MOMENTUM"):
        continue
    kv = dict(re.findall(r"(\w+)=(\S+)", line))
    key = (kv["mode"], int(kv["np"]), int(kv["thr"]))
    if "ERROR" in line:
        errors[key] += 1
        continue
    rows[key].append(kv)
modes = sys.argv[2:] or sorted({k[0] for k in rows})
def cell(vals):
    if any(v == "n/a" for v in vals):
        return "n/a"
    if len(set(vals)) == 1:
        return f"{float(vals[0]):.1e} (=)"
    f = sorted(float(v) for v in vals)
    return f"{statistics.median(f):.1e} [{f[0]:.1e}–{f[-1]:.1e}]"
for m in modes:
    print(f"#### `{m}`\n")
    print("| np | thr | " + " | ".join(cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    for np_ in (1, 2, 4, 8):
        for thr in (1, 8):
            r = rows.get((m, np_, thr), [])
            note = f" ({errors[(m, np_, thr)]} timed out)" if errors[(m, np_, thr)] else ""
            if not r:
                print(f"| {np_} | {thr} | " + " | ".join(["—"] * len(cols)) + f" |{note}")
                continue
            print(f"| {np_} | {thr} | " + " | ".join(cell([x[c] for x in r]) for c in cols) + f" |{note}")
    print()
