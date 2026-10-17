"""Signal generation, windowing, FFT analysis, filtering and spectrogram (NumPy only)."""
import numpy as np

MAX_N = 16384
MIN_N = 64
WINDOWS = ("rectangular", "hann", "hamming", "blackman")

_MORSE = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.", "G": "--.", "H": "....",
    "I": "..", "J": ".---", "K": "-.-", "L": ".-..", "M": "--", "N": "-.", "O": "---", "P": ".--.",
    "Q": "--.-", "R": ".-.", "S": "...", "T": "-", "U": "..-", "V": "...-", "W": ".--", "X": "-..-",
    "Y": "-.--", "Z": "--..", "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..", "9": "----.",
}


# ----------------------------------------------------------------------
# Generators
# ----------------------------------------------------------------------
def _morse_keying(text, wpm, fs, n):
    """0/1 keying envelope for `text` (PARIS timing), padded or cut to n samples."""
    unit = 1.2 / wpm
    key = np.zeros(n)
    t = 0.0
    for wi, word in enumerate(text.upper().split()):
        if wi:
            t += 7 * unit
        letters = [c for c in word if c in _MORSE]
        for li, ch in enumerate(letters):
            if li:
                t += 3 * unit
            for si, sym in enumerate(_MORSE[ch]):
                if si:
                    t += unit
                d = (1 if sym == "." else 3) * unit
                a, b = int(t * fs), min(n, int((t + d) * fs))
                if a < n:
                    key[a:b] = 1.0
                t += d
    return key


def preset(name, fs, n, freq=100.0, noise=0.0, text="SOS", wpm=20.0, seed=1):
    """Generate n samples of a named test signal at sample rate fs."""
    t = np.arange(n) / fs
    w = 2 * np.pi * freq
    if name == "sine":
        x = np.sin(w * t)
    elif name == "two_tone":
        x = 0.8 * np.sin(w * t) + 0.5 * np.sin(2 * np.pi * 2.6 * freq * t)
    elif name == "square":
        x = np.sign(np.sin(w * t))
    elif name == "triangle":
        x = (2 / np.pi) * np.arcsin(np.sin(w * t))
    elif name == "sawtooth":
        x = 2 * ((freq * t) % 1.0) - 1
    elif name == "chirp":
        f1 = 0.45 * fs
        k = (f1 - freq) / (n / fs)
        x = np.sin(2 * np.pi * (freq * t + 0.5 * k * t ** 2))
    elif name == "am":
        x = (1 + 0.6 * np.sin(2 * np.pi * freq * 0.1 * t)) * np.sin(w * t) / 1.6
    elif name == "morse":
        x = _morse_keying(text, wpm, fs, n)
        # 5 ms raised-cosine edges so the keying does not click
        k = max(1, int(0.005 * fs))
        x = np.convolve(x, np.hanning(2 * k + 1) / np.hanning(2 * k + 1).sum(), mode="same")
        x = x * np.sin(w * t)
    else:
        raise ValueError(f"Unknown signal '{name}'.")
    if noise > 0:
        x = x + noise * np.random.default_rng(seed).standard_normal(n)
    return x


# ----------------------------------------------------------------------
# Windows and spectra
# ----------------------------------------------------------------------
def window(name, n):
    if name == "rectangular":
        return np.ones(n)
    if name == "hann":
        return np.hanning(n + 1)[:-1]      # periodic form, correct for spectral analysis
    if name == "hamming":
        k = np.arange(n)
        return 0.54 - 0.46 * np.cos(2 * np.pi * k / n)
    if name == "blackman":
        k = np.arange(n)
        return 0.42 - 0.5 * np.cos(2 * np.pi * k / n) + 0.08 * np.cos(4 * np.pi * k / n)
    raise ValueError(f"Unknown window '{name}'.")


def amplitude_spectrum(x, win):
    """Single-sided amplitude spectrum, scaled so a sine of amplitude A peaks at A."""
    n = len(x)
    X = np.fft.rfft(x * win)
    cg = win.sum() / n                       # coherent gain of the window
    amp = np.abs(X) / (n * cg)
    amp[1:] *= 2
    if n % 2 == 0:
        amp[-1] /= 2                         # Nyquist bin is not doubled
    return amp


def find_peaks(freqs, amp, count=5, floor_ratio=0.02):
    """Top local maxima with parabolic interpolation of the true peak position."""
    if len(amp) < 3 or amp.max() < 1e-6:
        return []
    mid = amp[1:-1]
    is_peak = (mid > amp[:-2]) & (mid >= amp[2:]) & (mid > floor_ratio * amp.max())
    idx = np.nonzero(is_peak)[0] + 1
    idx = idx[np.argsort(amp[idx])[::-1]][:count]
    df = freqs[1] - freqs[0]
    out = []
    for i in idx:
        a, b, c = amp[i - 1], amp[i], amp[i + 1]
        den = a - 2 * b + c
        d = 0.5 * (a - c) / den if den != 0 else 0.0
        out.append({"freq": float(freqs[i] + d * df),
                    "amp": float(b - 0.25 * (a - c) * d)})
    return sorted(out, key=lambda p: p["freq"])


# ----------------------------------------------------------------------
# Filtering (zero-phase Butterworth magnitude applied in the frequency domain)
# ----------------------------------------------------------------------
FILTERS = ("lowpass", "highpass", "bandpass", "bandstop")


def response(kind, f, fc, fc2, order):
    """Butterworth-shaped magnitude response |H(f)|."""
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return _response(kind, f, fc, fc2, order)


def _response(kind, f, fc, fc2, order):
    f = np.asarray(f, float)
    eps = 1e-12
    if kind == "lowpass":
        return 1.0 / np.sqrt(1 + (f / fc) ** (2 * order))
    if kind == "highpass":
        r = (fc / np.maximum(f, eps)) ** (2 * order)
        return 1.0 / np.sqrt(1 + r)
    lo, hi = sorted((fc, fc2))
    f0 = np.sqrt(lo * hi)                    # geometric centre
    bw = hi - lo
    u = (f ** 2 - f0 ** 2) / np.maximum(f * bw, eps)    # low-pass to band mapping
    if kind == "bandpass":
        return 1.0 / np.sqrt(1 + u ** (2 * order))
    if kind == "bandstop":
        return 1.0 / np.sqrt(1 + (1.0 / np.maximum(np.abs(u), eps)) ** (2 * order))
    raise ValueError(f"Unknown filter '{kind}'.")


def apply_filter(x, fs, kind, fc, fc2, order):
    """Zero-phase filtering; zero-pads to 2N so circular wrap-around stays out of the result."""
    n = len(x)
    m = 1 << int(np.ceil(np.log2(2 * n)))
    X = np.fft.rfft(x, m)
    f = np.fft.rfftfreq(m, 1 / fs)
    return np.fft.irfft(X * response(kind, f, fc, fc2, order), m)[:n]


# ----------------------------------------------------------------------
# Spectrogram
# ----------------------------------------------------------------------
def spectrogram(x, fs, max_freq_bins=96, max_frames=200, floor_db=-80.0):
    n = len(x)
    nper = int(min(256, max(32, 1 << int(np.log2(max(32, n // 8))))))
    hop = max(1, nper // 4)
    if n < nper:
        return {"db": [], "t": [], "f": []}
    w = np.hanning(nper)
    starts = np.arange(0, n - nper + 1, hop)
    if len(starts) > max_frames:
        starts = starts[np.linspace(0, len(starts) - 1, max_frames).astype(int)]
    frames = np.stack([x[s:s + nper] * w for s in starts])
    S = np.abs(np.fft.rfft(frames, axis=1)) / (w.sum() / 2)
    f = np.fft.rfftfreq(nper, 1 / fs)
    if S.shape[1] > max_freq_bins:                          # keep peak energy per merged bin
        edges = np.linspace(0, S.shape[1], max_freq_bins + 1).astype(int)
        S = np.stack([S[:, a:max(b, a + 1)].max(axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=1)
        f = np.array([f[a] for a in edges[:-1]])
    # dB relative to amplitude 1 (same reference as the spectrum plot), so a quiet or empty
    # output looks quiet instead of being stretched to full scale
    db = np.clip(20 * np.log10(np.maximum(S, 1e-12)), floor_db, 0)
    return {"db": np.round(db, 0).astype(int).tolist(),
            "t": ((starts + nper / 2) / fs).tolist(), "f": f.tolist(), "floor_db": floor_db}


# ----------------------------------------------------------------------
# One call that does everything the UI needs
# ----------------------------------------------------------------------
def analyze(x, fs, win_name="hann", flt=None):
    x = np.asarray(x, float)
    n = len(x)
    if not MIN_N <= n <= MAX_N:
        raise ValueError(f"Signal length must be between {MIN_N} and {MAX_N} samples.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Signal contains non-finite values.")
    if win_name not in WINDOWS:
        raise ValueError(f"Unknown window '{win_name}'.")
    win = window(win_name, n)
    freqs = np.fft.rfftfreq(n, 1 / fs)

    y = x
    resp = None
    if flt:
        kind = flt.get("type")
        if kind not in FILTERS:
            raise ValueError(f"Unknown filter '{kind}'.")
        nyq = fs / 2
        fc = float(flt.get("fc", nyq / 4))
        fc2 = float(flt.get("fc2", fc * 2))
        fc2 = fc2 if kind in ("bandpass", "bandstop") else fc
        order = int(flt.get("order", 4))
        if kind in ("bandpass", "bandstop") and abs(fc2 - fc) < 0.01 * max(fc, fc2):
            raise ValueError("For a band filter the two cutoffs must differ by at least 1 percent.")
        band = kind in ("bandpass", "bandstop")
        if not (0 < fc < nyq and (not band or 0 < fc2 < nyq)):
            raise ValueError(f"Cutoff frequencies must be between 0 and {nyq:g} Hz.")
        if not 1 <= order <= 10:
            raise ValueError("Filter order must be between 1 and 10.")
        y = apply_filter(x, fs, kind, fc, fc2, order)
        resp = response(kind, freqs, fc, fc2, order)

    a_in = amplitude_spectrum(x, win)
    a_out = amplitude_spectrum(y, win)
    return {
        "y": y.tolist(),
        "freqs": freqs.tolist(),
        "amp_in": a_in.tolist(),
        "amp_out": a_out.tolist(),
        "response": None if resp is None else resp.tolist(),
        "peaks_in": find_peaks(freqs, a_in),
        "peaks_out": find_peaks(freqs, a_out),
        "rms_in": float(np.sqrt(np.mean(x ** 2))),
        "rms_out": float(np.sqrt(np.mean(y ** 2))),
        "resolution_hz": float(fs / n),
        "nyquist_hz": float(fs / 2),
        "spectrogram": spectrogram(y, fs),
    }
