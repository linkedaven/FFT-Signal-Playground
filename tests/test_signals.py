import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import signals as s

FS, N = 8000, 4096


def test_amplitude_calibration_all_windows():
    f = FS / N * 100                      # exactly on a bin
    x = 0.7 * np.sin(2 * np.pi * f * np.arange(N) / FS)
    for w in s.WINDOWS:
        assert abs(s.amplitude_spectrum(x, s.window(w, N)).max() - 0.7) < 1e-9


def test_parabolic_peak_interpolation():
    x = np.sin(2 * np.pi * 1000.37 * np.arange(N) / FS)
    freqs = np.fft.rfftfreq(N, 1 / FS)
    peak = s.find_peaks(freqs, s.amplitude_spectrum(x, s.window("hann", N)))[0]
    assert abs(peak["freq"] - 1000.37) < 0.25


def test_filters_pass_and_reject():
    t = np.arange(N) / FS
    x = np.sin(2 * np.pi * 200 * t) + np.sin(2 * np.pi * 2000 * t)
    rms = lambda v: np.sqrt(np.mean(v ** 2))
    lo = s.apply_filter(x, FS, "lowpass", 600, 0, 4)
    hi = s.apply_filter(x, FS, "highpass", 600, 0, 4)
    assert abs(rms(lo) - 1 / np.sqrt(2)) < 0.05 and abs(rms(hi) - 1 / np.sqrt(2)) < 0.05
    notch = s.apply_filter(x, FS, "bandstop", 150, 260, 4)
    assert abs(rms(notch) - 1 / np.sqrt(2)) < 0.05


def test_minus_3db_at_cutoff():
    assert abs(s.response("lowpass", [1000], 1000, 0, 4)[0] - 2 ** -0.5) < 1e-12
    assert abs(s.response("highpass", [1000], 1000, 0, 4)[0] - 2 ** -0.5) < 1e-12
    assert abs(s.response("bandpass", [200], 200, 800, 3)[0] - 2 ** -0.5) < 1e-12


def test_aliasing():
    x = s.preset("sine", 1000, 4096, freq=935)
    r = s.analyze(x, 1000)
    assert abs(r["peaks_in"][0]["freq"] - 65) < 1.0


def test_validation():
    x = np.zeros(N)
    for bad in [{"type": "lowpass", "fc": 5000}, {"type": "wat", "fc": 100},
                {"type": "bandpass", "fc": 500, "fc2": 500}, {"type": "lowpass", "fc": 100, "order": 99}]:
        try:
            s.analyze(x, FS, "hann", bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_empty_output_is_quiet():
    x = s.preset("sine", FS, N, freq=200)
    r = s.analyze(x, FS, "hann", {"type": "highpass", "fc": 3900 * 0.99, "order": 10})
    assert r["peaks_out"] == [] or r["peaks_out"][0]["amp"] < 1e-3
    assert max(max(row) for row in r["spectrogram"]["db"]) <= -60
