import numpy as np
from scipy.signal import welch

def bandpower_welch(x, fs, fmin, fmax):
    f, Pxx = welch(x, fs=fs, nperseg=min(1024, len(x)))
    m = (f >= fmin) & (f <= fmax)
    return float(np.trapz(Pxx[m], f[m]))

def norm01(x):
    x = np.asarray(x)
    return (x - x.min()) / (x.max() - x.min() + 1e-12)
