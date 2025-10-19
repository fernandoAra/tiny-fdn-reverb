import sys, numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

fs, x = wavfile.read(sys.argv[1])
x = x.astype(np.float32)
if x.ndim == 2: x = x.mean(axis=1)
x /= max(1e-9, np.abs(x).max())

edc = np.cumsum(x[::-1]**2)[::-1]
edc_db = 10*np.log10(np.maximum(edc, 1e-20))
edc_db -= edc_db[0]
t = np.arange(len(x))/fs

mask = (edc_db <= -5) & (edc_db >= -35)
p = np.polyfit(t[mask], edc_db[mask], 1) if mask.any() else [np.nan, np.nan]
rt60 = (-60.0/p[0]) if p[0] < 0 else np.nan

plt.figure(figsize=(6,3))
plt.plot(t, edc_db, label="EDC (dB)")
if p[0] == p[0]:
    tt = np.array([t[mask].min(), t[mask].max()])
    plt.plot(tt, np.polyval(p, tt), '--', label=f"Fit → RT60≈{rt60:.2f}s")
plt.axhline(-5, color='0.7', lw=0.5); plt.axhline(-35, color='0.7', lw=0.5)
plt.xlabel("Time (s)"); plt.ylabel("Level (dB)")
plt.title("Energy Decay Curve (Tiny FDN)")
plt.legend(); plt.tight_layout()
plt.savefig("eval/figs/fig_edc_rt60.png", dpi=150)
print("Saved eval/figs/fig_edc_rt60.png")
