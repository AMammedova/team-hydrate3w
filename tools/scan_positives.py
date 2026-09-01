import pyarrow.parquet as pq, numpy as np, collections, re
from pathlib import Path
root = Path("data/3W/dataset/9")
rows = []
for p in sorted(root.glob("WELL-*.parquet")):
    w = re.match(r'(WELL-\d+)_', p.name).group(1)
    s = pq.read_table(p, columns=["class"]).column("class").to_pandas()
    n = len(s)
    nan = int(s.isna().sum())
    v = s.dropna().astype("int64").to_numpy()
    rows.append(dict(well=w, file=p.name, n=n, nan_pct=100*nan/n,
                     n109=int((v==109).sum()), n9=int((v==9).sum())))

t_wells = sorted({r["well"] for r in rows if r["n109"]})
b_wells = sorted({r["well"] for r in rows if r["n9"]})
print(f"TRANSIENT (109) olan instans: {sum(1 for r in rows if r['n109'])}/57 -> {len(t_wells)} quyu")
print("   quyular:", t_wells)
print(f"BLOKAJ    (9)  olan instans: {sum(1 for r in rows if r['n9'])}/57 -> {len(b_wells)} quyu")
print("   quyular:", b_wells)
print()
print("Transient hadise sayi / quyu:", dict(collections.Counter(r["well"] for r in rows if r["n109"])))
print()
print("--- 43 'hadisesiz' instans (ne 109, ne 9) ---")
none_r = [r for r in rows if not r["n109"] and not r["n9"]]
nanp = np.array([r["nan_pct"] for r in none_r])
print(f"  sayi={len(none_r)}  NaN faizi: median={np.median(nanp):.1f}%  min={nanp.min():.1f}%  max={nanp.max():.1f}%")
print(f"  bu instanslardaki quyular: {sorted({r['well'] for r in none_r})}")
print()
print("--- butun 57 instans uzre NaN ---")
allnan = np.array([r["nan_pct"] for r in rows])
print(f"  NaN faizi: median={np.median(allnan):.1f}%  max={allnan.max():.1f}%  "
      f"tam-etiketli (NaN<1%) instans: {(allnan<1).sum()}/57")
print()
print("--- 14 transient instansin detali ---")
for r in rows:
    if r["n109"]:
        print(f"  {r['file']:<42} n={r['n']:>7}  109={r['n109']:>6}s  9={r['n9']:>6}s  NaN={r['nan_pct']:.1f}%")
