"""Setup-cost estimate for the PHRASE counts: distinct trigram / 4-gram keys and counting time on TRAIN prefixes
(numpy only, 1 thread; the harness runs setup with OMP_NUM_THREADS=1)."""
import time
import numpy as np
V = 2000
s = np.memmap("/Users/svi/Desktop/trlm/data/tokens/train.bin", dtype=np.uint16, mode="r")
for N in (25_000_000, 50_000_000, 100_000_000):
    x = np.asarray(s[:N], dtype=np.int64)
    t0 = time.time()
    k3 = (x[:-2] * V + x[1:-1]) * V + x[2:]
    u3, c3 = np.unique(k3, return_counts=True)
    t3 = time.time() - t0
    ctx3 = np.unique(u3 // V)
    keep3 = c3 >= 2
    t0 = time.time()
    k4 = ((x[:-3] * V + x[1:-2]) * V + x[2:-1]) * V + x[3:]
    u4, c4 = np.unique(k4, return_counts=True)
    t4 = time.time() - t0
    print(f"N={N/1e6:.0f}M: trigram entries {len(u3)/1e6:.2f}M (count>=2: {keep3.sum()/1e6:.2f}M), contexts {len(ctx3)/1e6:.2f}M, "
          f"{t3:.1f}s | 4-gram entries {len(u4)/1e6:.2f}M (count>=2: {(c4 >= 2).sum()/1e6:.2f}M), {t4:.1f}s", flush=True)
    del x, k3, u3, c3, k4, u4, c4
