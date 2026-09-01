import pyarrow.parquet as pq, numpy as np, collections
from pathlib import Path
root = Path("data/3W/dataset/9")

tbl = pq.read_table(sorted(root.glob("WELL-*.parquet"))[0], columns=["class"])
print("class sutununun tipi:", tbl.schema.field("class").type)

glob_cnt = collections.Counter()
per_inst = collections.Counter()
for p in sorted(root.glob("WELL-*.parquet")):
    arr = pq.read_table(p, columns=["class"]).column("class").to_pandas()
    vals = arr.dropna().astype("int64").to_numpy()
    glob_cnt.update(collections.Counter(vals))
    per_inst.update({v: 1 for v in set(vals.tolist())})
    if arr.isna().any(): per_inst.update({"NaN": 1})

print("\nREAL Event-9 (57 instans) -- butun etiket kodlari:")
for k, v in sorted(glob_cnt.items()):
    print(f"  kod {k:>4}: {v:>10,} musahide   ({per_inst[k]:>2}/57 instansda var)")
print(f"  NaN olan instans sayi: {per_inst.get('NaN',0)}/57")

print("\nSIMULATED (150 instans) -- etiket kodlari:")
gs, pi = collections.Counter(), collections.Counter()
for p in sorted(root.glob("SIMULATED*.parquet")):
    arr = pq.read_table(p, columns=["class"]).column("class").to_pandas()
    vals = arr.dropna().astype("int64").to_numpy()
    gs.update(collections.Counter(vals)); pi.update({v: 1 for v in set(vals.tolist())})
for k, v in sorted(gs.items()):
    print(f"  kod {k:>4}: {v:>10,} musahide   ({pi[k]:>3}/150 instansda var)")
