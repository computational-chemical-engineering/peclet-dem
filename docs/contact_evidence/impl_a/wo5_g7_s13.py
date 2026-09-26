"""usage: python wo5_g7_s13.py <wo5_g7_s13.sh output> <outdir> : G7a/c/e/f as restated by §12 S13 / S18."""
import re, sys, numpy as np
log, out = sys.argv[1], sys.argv[2]
blocks, cur = {}, None
for l in open(log):
    m = re.match(r'== (\S+) (.*)', l)
    if m:
        cur = (m.group(1), m.group(2).strip()); blocks[cur] = []
    elif cur:
        blocks[cur].append(l.strip())
def field(lines, tag, key):
    for l in lines:
        if l.startswith(tag):
            m = re.search(r'\b' + key + r'=(\S+)', l)
            if m: return float(m.group(1))
def ke_last(lines):
    for l in lines:
        if l.startswith('KE'):
            return float(l.split()[-1].split('=')[1])
def load(p):
    a = np.fromfile(p, dtype=np.uint32).reshape(-1, 14)
    f = a.view(np.float32)
    return a[:, 0].view(np.int32), f[:, 4:7]
print("G7a tri_pgs --axis=2 --no-stop: it, KE_np1, KE_np2, |dKE|/KE_np1, KE_np2 <= KE_np1 (1+1e-6); <= 1e-3 at 16, <= 1e-5 at 64")
prev = None
for it in (4, 8, 16, 32, 64):
    k1 = ke_last(blocks[('G7a', f'it={it} np=1')]); k2 = ke_last(blocks[('G7a', f'it={it} np=2')])
    r = abs(k2 - k1) / k1
    lim = {16: 1e-3, 64: 1e-5}.get(it)
    print(f"  {it:3d} {k1:.7g} {k2:.7g} {r:.3e} {k2 <= k1 * (1 + 1e-6)}" + ("" if prev is None or r <= prev else "  (INCREASED)")
          + ("" if lim is None else (" ok" if r <= lim else " HARD-FAIL")))
    prev = r
print("G7c cluster_pgs --steps=1 --no-stop: it 8/32/128/256, RMS |v_np - v_np1(same N)| / |v|; non-increasing, <= 1e-3 at 256")
for np_ in (2, 4, 8):
    row = []
    for it in (8, 32, 128, 256):
        g1, v1 = load(f"{out}/g7c_it{it}_np1.dump"); g, v = load(f"{out}/g7c_it{it}_np{np_}.dump")
        assert (g1 == g).all()
        rel = np.linalg.norm(v - v1, axis=1) / np.maximum(np.linalg.norm(v1, axis=1), 1e-30)
        row.append(float(np.sqrt(np.mean(rel ** 2))))
    print(f"  np {np_}: " + " ".join(f"{x:.3e}" for x in row) + ("" if all(a >= b for a, b in zip(row, row[1:])) else "  (NOT non-increasing)")
          + (" ok" if row[-1] <= 1e-3 else " HARD-FAIL"))
print("G7e ovl (max over steps), ratio to np 1; hard <= 3 x np1 (S18), ratio reported")
for m in ('cluster', 'cluster_pgs', 'cluster_friction'):
    o1 = field(blocks[('G7e', f'{m} np=1')], 'MOMENTUM', 'ovl')
    s = f"  {m}: np1 {o1:.3e}"
    for np_ in (2, 4, 8):
        o = field(blocks[('G7e', f'{m} np={np_}')], 'MOMENTUM', 'ovl')
        hard = o <= 3 * o1
        s += f" | np{np_} {o:.3e} ({o / o1:.2f}x{'' if hard else ' HARD-FAIL'})"
    print(s)
print("G7f ITERS ratio npN / np1 (hard: np2 <= 3.5, np4/8 <= 5)")
for kind, key in (('pos', 'pos'), ('vel', 'vel')):
    i1 = field(blocks[('G7f', f'{kind} np=1')], 'ITERS', key)
    s = f"  {kind}: np1 {int(i1)}"
    for np_ in (2, 4, 8):
        i = field(blocks[('G7f', f'{kind} np={np_}')], 'ITERS', key)
        lim = 3.5 if np_ == 2 else 5.0
        s += f" | np{np_} {int(i)} ({i / i1:.2f}x{'' if i / i1 <= lim else ' HARD-FAIL'})"
    print(s)
