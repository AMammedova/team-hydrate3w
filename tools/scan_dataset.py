import pyarrow.parquet as pq, numpy as np, collections, re
from pathlib import Path
root = Path("data/3W/dataset")

def wid(p):
    m = re.match(r'(WELL-\d+)_', p.name)
    return m.group(1) if m else ("SIM" if p.name.startswith("SIMULATED") else "DRAWN")

# --- 1) timestep counts via metadata only (fast) ---
for cls in ("9", "0"):
    rows = []
    for p in sorted((root/cls).glob("*.parquet")):
        n = pq.ParquetFile(p).metadata.num_rows
        rows.append((wid(p), n))
    real = [n for w,n in rows if w.startswith("WELL")]
    sim  = [n for w,n in rows if w == "SIM"]
    print(f"--- class {cls} ---")
    if real:
        a = np.array(real)
        print(f"  REAL n={len(a)}  timesteps: median={int(np.median(a))}  min={a.min()}  max={a.max()}  "
              f"cem saat={a.sum()/3600:.1f}")
        print(f"  >= 10k timestep sertini kecen: {(a>=10000).sum()}/{len(a)}")
    if sim:
        b = np.array(sim)
        print(f"  SIM  n={len(b)}  timesteps: median={int(np.median(b))}  min={b.min()}  max={b.max()}")
    if cls == "0":
        per = collections.defaultdict(float)
        for w,n in rows: per[w] += n/3600
        print("  NORMAL saat / quyu:", {k: round(v,1) for k,v in sorted(per.items())})
        print(f"  CEM normal saat: {sum(per.values()):.1f}")

# --- 2) transient duration + monotonicity on the 57 real Event-9 instances ---
print("\n--- EVENT 9 real: transient faza (label 109) ---")
durs, mono_ok, no_trans, no_block = [], 0, 0, 0
for p in sorted((root/"9").glob("WELL-*.parquet")):
    lab = pq.read_table(p, columns=["class"]).column("class").to_numpy(zero_copy_only=False)
    lab = np.nan_to_num(lab.astype("float64"), nan=-1).astype(int)
    n109 = int((lab == 109).sum()); n9 = int((lab == 9).sum())
    if n109 == 0: no_trans += 1
    if n9 == 0:   no_block += 1
    if n109: durs.append(n109)
    sev = np.select([lab == 109, lab == 9], [1, 2], default=0)
    if np.all(np.diff(sev) >= 0): mono_ok += 1
d = np.array(durs)
print(f"  instans: 57 | transient fazasi olan: {len(d)} | transient YOXDUR: {no_trans} | blokaj YOXDUR: {no_block}")
print(f"  transient davametme (saniye): median={int(np.median(d))} min={d.min()} max={d.max()} "
      f"q25={int(np.percentile(d,25))} q75={int(np.percentile(d,75))}")
print(f"  60s pencereden qisa olan transientler: {(d<60).sum()}/{len(d)}")
print(f"  monoton severity (is_monotonic_severity==True): {mono_ok}/57")
