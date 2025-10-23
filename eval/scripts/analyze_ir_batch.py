import sys, os, math, numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

def load_mono_norm(path):
    fs, x = wavfile.read(path)
    x = x.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    # trim leading silence (threshold -60 dB)
    thr = 10**(-60/20)
    start = 0
    for i in range(min(len(x), int(0.5*fs))):
        if abs(x[i]) > thr:
            start = i; break
    x = x[start:]
    # normalize to unity peak (avoid div0)
    peak = max(1e-9, float(np.max(np.abs(x))))
    x = x / peak
    return fs, x

def schroeder_edc(x):
    edc = np.cumsum(x[::-1]**2)[::-1]
    edc = np.maximum(edc, 1e-20)
    edc_db = 10*np.log10(edc)
    edc_db -= edc_db[0]
    return edc_db

def fit_rt60(t, edc_db, lo=-35.0, hi=-5.0):
    mask = (edc_db <= hi) & (edc_db >= lo)
    if not np.any(mask):
        return np.nan, np.nan, None
    p = np.polyfit(t[mask], edc_db[mask], 1)  # dB vs s
    slope, intercept = p[0], p[1]
    if slope >= 0:
        return np.nan, np.nan, (slope, intercept, mask)
    rt60 = -60.0 / slope
    edt  = -10.0 / slope * 1000.0  # ms
    return rt60, edt, (slope, intercept, mask)

def echo_density(x, fs, win_ms=10, step_ms=5, thr_rel=0.02):
    win = max(8, int((win_ms/1000.0)*fs))
    step = max(1, int((step_ms/1000.0)*fs))
    # simple peak detector (strict local maxima) with relative threshold
    absmax = max(1e-6, float(np.max(np.abs(x))))
    thr = thr_rel * absmax
    peaks = np.r_[False, (x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]), False] & (np.abs(x) > thr)
    idxs = range(0, len(x)-win, step)
    times = np.array([(i+win/2)/fs for i in idxs], dtype=np.float32)
    density = np.array([ peaks[i:i+win].sum() * (fs/win) for i in idxs ], dtype=np.float32)  # peaks/s
    return times, density

def spectral_centroid(x, fs, hop_ms=10, win_ms=40):
    hop = max(1, int((hop_ms/1000.0)*fs))
    win = max(hop, int((win_ms/1000.0)*fs))
    w = np.hanning(win)
    cents_t, cents = [], []
    i = 0
    while i+win <= len(x):
        seg = x[i:i+win] * w
        spec = np.abs(np.fft.rfft(seg))
        freqs = np.fft.rfftfreq(win, d=1/fs)
        s = spec.sum()
        c = (freqs @ spec) / s if s > 1e-12 else 0.0
        cents_t.append((i+win/2)/fs)
        cents.append(c)
        i += hop
    return np.array(cents_t), np.array(cents)

def analyze_one(path):
    fs, x = load_mono_norm(path)
    t = np.arange(len(x))/fs
    edc_db = schroeder_edc(x)
    rt60, edt_ms, fitinfo = fit_rt60(t, edc_db, lo=-35, hi=-5)
    times_d, dens = echo_density(x, fs, win_ms=10, step_ms=5, thr_rel=0.02)
    t_sc, sc = spectral_centroid(x, fs, hop_ms=10, win_ms=40)
    return dict(fs=fs, t=t, x=x, edc_db=edc_db, rt60=rt60, edt_ms=edt_ms,
                fitinfo=fitinfo, times_d=times_d, dens=dens, t_sc=t_sc, sc=sc)

def label_from_filename(fn):
    base = os.path.basename(fn).lower()
    m = "Hadamard" if "had" in base else ("Householder" if "house" in base else "UnknownM")
    d = "Spread" if "spread" in base else ("Prime" if "prime" in base else "UnknownD")
    return f"{m}-{d}"

def main(args):
    if len(args) < 2:
        print("Usage: python analyze_ir_batch.py IR1.wav IR2.wav [IR3.wav ...]")
        sys.exit(1)
    os.makedirs("eval/figs", exist_ok=True)

    results = []
    for path in args[1:]:
        res = analyze_one(path)
        res["path"] = path
        res["label"] = label_from_filename(path)
        results.append(res)

    # ---- Summary CSV ----
    with open("eval/figs/summary.csv", "w") as f:
        f.write("file,label,rt60_s,edt_ms,peak_density_100ms_s,mean_density_s\n")
        for r in results:
            # simple density stats (peak in first 300ms window region)
            sel = r["times_d"] <= 0.3
            peak_100 = float(np.max(r["dens"][sel])) if np.any(sel) else float(np.max(r["dens"]))
            f.write(f'{r["path"]},{r["label"]},{r["rt60"]:.4f},{r["edt_ms"]:.2f},{peak_100:.2f},{float(np.mean(r["dens"])):.2f}\n')

    # ---- Combined plots ----
    # 1) EDC overlays + fit lines + legend
    plt.figure(figsize=(8,4))
    for r in results:
        plt.plot(r["t"], r["edc_db"], label=f'{r["label"]}  (RT60≈{r["rt60"]:.2f}s, EDT≈{r["edt_ms"]:.0f}ms)')
        if r["fitinfo"] is not None:
            slope, intercept, mask = r["fitinfo"]
            tt = np.array([r["t"][mask].min(), r["t"][mask].max()])
            plt.plot(tt, slope*tt + intercept, "--")
    plt.axhline(-5, color="0.8", lw=0.8); plt.axhline(-35, color="0.8", lw=0.8)
    plt.xlabel("Time (s)"); plt.ylabel("EDC (dB)"); plt.title("Energy Decay Curves (Schroeder)")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig("eval/figs/edc_overlay.png", dpi=150)

    # 2) Echo density (events/s) overlays
    plt.figure(figsize=(8,3))
    for r in results:
        plt.plot(r["times_d"], r["dens"], label=r["label"])
    plt.xlabel("Time (s)"); plt.ylabel("Peaks per second"); plt.title("Echo density (10 ms window)")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig("eval/figs/density_overlay.png", dpi=150)

    # 3) Spectral centroid (perceptual brightness proxy)
    plt.figure(figsize=(8,3))
    for r in results:
        plt.plot(r["t_sc"], r["sc"], label=r["label"])
    plt.xlabel("Time (s)"); plt.ylabel("Spectral centroid (Hz)"); plt.title("Brightness vs time")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig("eval/figs/centroid_overlay.png", dpi=150)

    print("Wrote:")
    print("  eval/figs/summary.csv")
    print("  eval/figs/edc_overlay.png")
    print("  eval/figs/density_overlay.png")
    print("  eval/figs/centroid_overlay.png")

if __name__ == "__main__":
    main(sys.argv)
