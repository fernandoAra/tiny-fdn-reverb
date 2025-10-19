import sys, numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

fs, x = wavfile.read(sys.argv[1])
x = x.astype(np.float32)
if x.ndim == 2: x = x.mean(axis=1)
x /= max(1e-9, np.abs(x).max())

win = int(0.010*fs)
peaks = np.r_[False, (x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]), False] & (np.abs(x) > 0.02)
step = win//2
idxs = range(0, len(x)-win, step)
times = np.array([ (i+win/2)/fs for i in idxs ])
dens  = np.array([ peaks[i:i+win].sum() * (fs/win) for i in idxs ])

plt.figure(figsize=(6,3))
plt.plot(times, dens)
plt.xlabel("Time (s)"); plt.ylabel("Peaks per second")
plt.title("Echo density (10 ms window)")
plt.tight_layout()
plt.savefig("eval/figs/fig_density.png", dpi=150)
print("Saved eval/figs/fig_density.png")
