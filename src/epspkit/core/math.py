import numpy as np
from scipy.signal import (
    savgol_filter,
    butter,
    filtfilt,
    find_peaks as _find_peaks,
    peak_prominences as _peak_prominences,
)
from scipy.ndimage import uniform_filter1d


def gradient(y: np.ndarray, x: np.ndarray):
    return np.gradient(y, x)

def auc(x: np.ndarray, y: np.ndarray):
    return np.trapz(y, x)

def logistic(x, A, x0, k):
    return A / (1 + np.exp(-(x - x0) / k))

def exp_saturation(x, A, k):
    return A * (1 - np.exp(-k * x))

def baseline(y):
    return np.mean(y), np.std(y)

def moving_average(y: np.ndarray, window_size: int):
    return uniform_filter1d(y, size=window_size, mode='nearest')

def rms(y: np.ndarray):
    return np.sqrt(np.mean(y**2))

def savgol(y: np.ndarray, window_size: int, polyorder: int):
    return savgol_filter(y, window_size, polyorder)

def butter_lowpass(y: np.ndarray, cutoff: float, fs: float, order: int = 3):
    b, a = butter(order, cutoff, btype='low', fs=fs)
    return filtfilt(b, a, y)

def linear_fit(x: np.ndarray, y: np.ndarray):
    m, b = np.polyfit(x, y, 1)
    y_fit = m * x + b
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(m), float(b), float(r2)

def find_peaks(y: np.ndarray, **kwargs):
    return _find_peaks(y, **kwargs)

def peak_prominences(y: np.ndarray, peaks: np.ndarray):
    return _peak_prominences(y, peaks)

def to_samples(time_ms: float, fs: float):
    return int(round((time_ms / 1000.0) * fs))

def to_ms(samples: int, fs: float):
    return (samples / fs) * 1000.0
