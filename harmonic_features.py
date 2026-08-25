#!/usr/bin/env python
"""Acoustic/harmonic feature extraction for sarcasm detection.

Computes, per utterance audio file:
  - Prosody:      F0 mean/std/range, jitter, shimmer, HNR (via Praat/parselmouth)
  - Rhythm:       voiced ratio, speaking rate proxy, pause statistics
  - Harmony:      sensory dissonance (Sethares 1993) and inharmonicity,
                  both computed from FFT spectral peaks frame-by-frame.

No essentia dependency (no macOS/py3.14 wheel available); dissonance/inharmonicity
are implemented directly from FFT peaks with numpy/scipy.
"""
import numpy as np
import parselmouth
from scipy.signal import find_peaks

# --- Sethares (1993) sensory dissonance model -------------------------------
# "Local Consonance and the Relationship Between Timbre and Scale", J. Acoust.
# Soc. Am. Constants below are the ones from the widely reproduced dissmeasure.m.
_D_STAR = 0.24
_S1 = 0.0207
_S2 = 18.96
_C1, _C2 = 5.0, -5.0
_A1, _A2 = -3.51, -5.75


def sethares_dissonance(freqs: np.ndarray, amps: np.ndarray) -> float:
    """Total pairwise sensory dissonance among a set of spectral peaks (partials).

    freqs, amps: 1-D arrays of peak frequency (Hz) and amplitude (linear).
    Higher value = rougher/more dissonant spectrum.
    """
    if len(freqs) < 2:
        return 0.0
    order = np.argsort(freqs)
    f = freqs[order]
    a = amps[order]

    total = 0.0
    n = len(f)
    for i in range(1, n):
        f_min = f[: n - i]
        f_max = f[i:]
        a_pair = a[: n - i] * a[i:]
        s = _D_STAR / (_S1 * f_min + _S2)
        f_diff = f_max - f_min
        loudness = _C1 * np.exp(_A1 * s * f_diff) + _C2 * np.exp(_A2 * s * f_diff)
        total += np.sum(a_pair * loudness)
    return float(total)


def inharmonicity(freqs: np.ndarray, amps: np.ndarray, f0: float) -> float:
    """Energy-weighted deviation of spectral peaks from the ideal harmonic series of f0.

    0 = perfectly harmonic (clean voiced tone); higher = noisier/breathier/rougher voice.
    """
    if f0 <= 0 or len(freqs) == 0:
        return 0.0
    harmonic_number = np.maximum(np.round(freqs / f0), 1)
    deviation = np.abs(freqs - harmonic_number * f0)
    weights = amps ** 2
    denom = f0 * np.sum(weights)
    if denom == 0:
        return 0.0
    return float(2 * np.sum(deviation * weights) / denom)


# --- Frame-level spectral peak picking --------------------------------------

def spectral_peaks(frame: np.ndarray, sr: int, n_peaks: int = 10,
                    min_freq: float = 60.0, max_freq: float = 5000.0):
    """Returns (freqs, amps) of the strongest spectral peaks in a windowed frame."""
    window = np.hanning(len(frame))
    spectrum = np.abs(np.fft.rfft(frame * window))
    freqs_axis = np.fft.rfftfreq(len(frame), d=1.0 / sr)

    band = (freqs_axis >= min_freq) & (freqs_axis <= max_freq)
    spectrum = spectrum.copy()
    spectrum[~band] = 0.0

    if spectrum.max() <= 0:
        return np.array([]), np.array([])

    peak_idx, _ = find_peaks(spectrum, height=spectrum.max() * 0.02)
    if len(peak_idx) == 0:
        return np.array([]), np.array([])

    peak_amps = spectrum[peak_idx]
    top = np.argsort(peak_amps)[::-1][:n_peaks]
    return freqs_axis[peak_idx][top], peak_amps[top]


# --- Prosody / rhythm (Praat via parselmouth) -------------------------------

def prosody_features(sound: parselmouth.Sound) -> dict:
    pitch = sound.to_pitch()
    f0 = pitch.selected_array["frequency"]
    voiced = f0[f0 > 0]

    point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 500)
    try:
        jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    except Exception:
        jitter = np.nan
    try:
        shimmer = parselmouth.praat.call(
            [sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    except Exception:
        shimmer = np.nan

    harmonicity = sound.to_harmonicity()
    hnr_values = harmonicity.values[harmonicity.values != -200]

    return {
        "f0_mean": float(np.mean(voiced)) if len(voiced) else np.nan,
        "f0_std": float(np.std(voiced)) if len(voiced) else np.nan,
        "f0_range": float(np.ptp(voiced)) if len(voiced) else np.nan,
        "jitter_local": float(jitter),
        "shimmer_local": float(shimmer),
        "hnr_mean": float(np.mean(hnr_values)) if len(hnr_values) else np.nan,
        "voiced_ratio": float(len(voiced) / len(f0)) if len(f0) else np.nan,
    }


def rhythm_features(sound: parselmouth.Sound) -> dict:
    intensity = sound.to_intensity()
    values = intensity.values[0]
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"speaking_rate_proxy": np.nan, "energy_std": np.nan}

    threshold = np.mean(values) - 3  # dB below mean marks a syllable-nucleus-ish peak region
    peaks, _ = find_peaks(values, height=threshold, distance=int(0.08 / intensity.time_step))
    duration = sound.get_total_duration()
    return {
        "speaking_rate_proxy": float(len(peaks) / duration) if duration > 0 else np.nan,
        "energy_std": float(np.std(values)),
    }


# --- Harmony features over the whole utterance ------------------------------

def harmony_features(sound: parselmouth.Sound, frame_length_s: float = 0.04,
                      hop_s: float = 0.02) -> dict:
    y = sound.values[0]
    sr = int(sound.sampling_frequency)

    pitch = sound.to_pitch(time_step=hop_s)
    f0_track = pitch.selected_array["frequency"]
    times = pitch.ts()

    frame_len = int(frame_length_s * sr)
    dissonances, inharmonicities = [], []

    for t, f0 in zip(times, f0_track):
        if f0 <= 0:
            continue
        center = int(t * sr)
        start, end = center - frame_len // 2, center + frame_len // 2
        if start < 0 or end > len(y):
            continue
        freqs, amps = spectral_peaks(y[start:end], sr)
        if len(freqs) < 2:
            continue
        dissonances.append(sethares_dissonance(freqs, amps))
        inharmonicities.append(inharmonicity(freqs, amps, f0))

    def stats(values, name):
        if not values:
            return {f"{name}_mean": np.nan, f"{name}_std": np.nan}
        return {f"{name}_mean": float(np.mean(values)), f"{name}_std": float(np.std(values))}

    return {**stats(dissonances, "dissonance"), **stats(inharmonicities, "inharmonicity")}


# --- Scale / key-fit features (Krumhansl-Schmuckler style) ------------------
# Standard key profiles from Krumhansl & Kessler (1982), the classic key-finding
# algorithm. Applying them to a speech F0 contour is an analogy, not literal
# tonal harmony: speech pitch glides continuously rather than landing on
# discrete scale degrees, so this measures how much the utterance's pitch-class
# distribution *resembles* a major/minor/whole-tone scale, not a real key.
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_WHOLETONE_PROFILE = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])


def pitch_class_histogram(f0_track: np.ndarray, ref_freq: float = 440.0) -> np.ndarray:
    """12-bin pitch-class histogram (a speech analogue of a music chroma vector)."""
    voiced = f0_track[f0_track > 0]
    if len(voiced) == 0:
        return np.zeros(12)
    midi = 69 + 12 * np.log2(voiced / ref_freq)
    pitch_classes = np.round(midi).astype(int) % 12
    hist = np.bincount(pitch_classes, minlength=12).astype(float)
    total = hist.sum()
    return hist / total if total > 0 else hist


def _best_rotation_correlation(histogram: np.ndarray, template: np.ndarray) -> float:
    """Max Pearson correlation between the histogram and the template over all 12 transpositions."""
    if np.std(histogram) == 0:
        return 0.0
    best = -1.0
    for shift in range(12):
        rotated = np.roll(template, shift)
        corr = np.corrcoef(histogram, rotated)[0, 1]
        best = max(best, corr)
    return float(best)


def scale_features(sound: parselmouth.Sound, hop_s: float = 0.02) -> dict:
    pitch = sound.to_pitch(time_step=hop_s)
    f0_track = pitch.selected_array["frequency"]
    histogram = pitch_class_histogram(f0_track)

    major_fit = _best_rotation_correlation(histogram, _MAJOR_PROFILE)
    minor_fit = _best_rotation_correlation(histogram, _MINOR_PROFILE)
    wholetone_fit = _best_rotation_correlation(histogram, _WHOLETONE_PROFILE)

    nonzero = histogram[histogram > 0]
    entropy = -np.sum(nonzero * np.log2(nonzero)) if len(nonzero) else 0.0
    tonal_clarity = 1 - entropy / np.log2(12)

    return {
        "major_fit": major_fit,
        "minor_fit": minor_fit,
        "wholetone_fit": wholetone_fit,
        "majorness": major_fit - minor_fit,
        "tonal_clarity": float(tonal_clarity),
    }


# --- Microtonal (fine-resolution) scale features ----------------------------
# Speech F0 glides continuously and rarely lands exactly on a 12-TET semitone,
# so hard-rounding to 1-of-12 bins injects boundary-quantization noise near
# every bin edge. These variants raise the resolution (e.g. 60 bins = a fifth-
# tone each) and rebuild the Krumhansl-Kessler templates at that resolution via
# circular Gaussian smoothing (a plain spline would ring on the non-smooth
# whole-tone step template).

def _smooth_template(profile_12: np.ndarray, n_bins: int, bandwidth: float = 0.5) -> np.ndarray:
    """Upsamples a 12-point-per-octave profile to n_bins via circular Gaussian smoothing."""
    positions = np.arange(n_bins) * 12.0 / n_bins
    degrees = np.arange(12)
    diff = np.abs(positions[:, None] - degrees[None, :])
    circular_dist = np.minimum(diff, 12 - diff)
    weights = np.exp(-0.5 * (circular_dist / bandwidth) ** 2)
    return weights @ profile_12


def pitch_class_histogram_fine(f0_track: np.ndarray, n_bins: int = 60, ref_freq: float = 440.0,
                                bandwidth: float = 0.5) -> np.ndarray:
    """Fine-resolution pitch-class density via circular Gaussian KDE (not hard binning).

    Hard-rounding each observation to 1-of-n_bins would make the empirical
    histogram a set of sharp spikes while the (necessarily smoothed, see
    _smooth_template) key templates are broad curves -- correlating a spiky
    signal against a smooth one deflates the fit even for a perfect match.
    Smoothing the observations with the *same* kernel as the templates keeps
    the comparison apples-to-apples.
    """
    voiced = f0_track[f0_track > 0]
    if len(voiced) == 0:
        return np.zeros(n_bins)
    midi = 69 + 12 * np.log2(voiced / ref_freq)
    pitch_pos = midi % 12

    bin_positions = np.arange(n_bins) * 12.0 / n_bins
    diff = np.abs(bin_positions[:, None] - pitch_pos[None, :])
    circular_dist = np.minimum(diff, 12 - diff)
    density = np.exp(-0.5 * (circular_dist / bandwidth) ** 2).sum(axis=1)
    total = density.sum()
    return density / total if total > 0 else density


def _best_rotation_correlation_fine(histogram: np.ndarray, template: np.ndarray) -> float:
    n_bins = len(histogram)
    if np.std(histogram) == 0:
        return 0.0
    best = -1.0
    for shift in range(n_bins):
        rotated = np.roll(template, shift)
        best = max(best, np.corrcoef(histogram, rotated)[0, 1])
    return float(best)


def scale_features_fine(sound: parselmouth.Sound, n_bins: int = 60, hop_s: float = 0.02,
                         bandwidth: float = 0.5) -> dict:
    pitch = sound.to_pitch(time_step=hop_s)
    f0_track = pitch.selected_array["frequency"]
    histogram = pitch_class_histogram_fine(f0_track, n_bins=n_bins, bandwidth=bandwidth)

    major_template = _smooth_template(_MAJOR_PROFILE, n_bins, bandwidth)
    minor_template = _smooth_template(_MINOR_PROFILE, n_bins, bandwidth)
    wholetone_template = _smooth_template(_WHOLETONE_PROFILE, n_bins, bandwidth)

    major_fit = _best_rotation_correlation_fine(histogram, major_template)
    minor_fit = _best_rotation_correlation_fine(histogram, minor_template)
    wholetone_fit = _best_rotation_correlation_fine(histogram, wholetone_template)

    nonzero = histogram[histogram > 0]
    entropy = -np.sum(nonzero * np.log2(nonzero)) if len(nonzero) else 0.0
    tonal_clarity = 1 - entropy / np.log2(n_bins)

    return {
        "major_fit_fine": major_fit,
        "minor_fit_fine": minor_fit,
        "wholetone_fit_fine": wholetone_fit,
        "majorness_fine": major_fit - minor_fit,
        "tonal_clarity_fine": float(tonal_clarity),
    }


def extract_all_features(path: str) -> dict:
    sound = parselmouth.Sound(path)
    if sound.get_number_of_channels() > 1:
        sound = sound.convert_to_mono()
    features = {}
    features.update(prosody_features(sound))
    features.update(rhythm_features(sound))
    features.update(harmony_features(sound))
    features.update(scale_features(sound))
    return features


if __name__ == "__main__":
    # Sanity check: dissonance should rank musical intervals the way music
    # theory predicts (unison/octave/fifth = consonant, minor 2nd/tritone = dissonant),
    # confirming the Sethares implementation before trusting it on speech.
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    base_freq = 220.0

    intervals = {
        "unison (1:1)": 1.0,
        "minor 2nd": 2 ** (1 / 12),
        "major 3rd": 2 ** (4 / 12),
        "perfect 5th": 2 ** (7 / 12),
        "tritone": 2 ** (6 / 12),
        "octave (2:1)": 2.0,
    }

    print(f"{'interval':<16}{'dissonance':>12}")
    for name, ratio in intervals.items():
        f2 = base_freq * ratio
        signal = np.sin(2 * np.pi * base_freq * t) + np.sin(2 * np.pi * f2 * t)
        freqs, amps = spectral_peaks(signal, sr, n_peaks=10)
        d = sethares_dissonance(freqs, amps)
        print(f"{name:<16}{d:>12.4f}")

    # Sanity check 2: uniform note *presence* can't separate major from natural
    # minor, because as unordered pitch-class sets they're rotations of the same
    # 7-note diatonic collection (every mode of a scale shares its note set).
    # What actually distinguishes them is emphasis: a melody leans on its tonic,
    # dominant, and third -- and the third is exactly what differs between
    # parallel major (major 3rd, pitch class 4) and minor (minor 3rd, pitch
    # class 3). So the test below weights degrees the way a real melody would.
    print()
    print(f"{'test melody':<18}{'major_fit':>10}{'minor_fit':>10}{'wholetone_fit':>14}")
    weighted_scales = {
        "C major-ish":   {0: 3, 4: 2, 7: 2, 2: 1, 5: 1, 9: 1, 11: 1},
        "C minor-ish":   {0: 3, 3: 2, 7: 2, 2: 1, 5: 1, 8: 1, 10: 1},
        "C whole-tone":  {0: 1, 2: 1, 4: 1, 6: 1, 8: 1, 10: 1},
    }
    for name, weights in weighted_scales.items():
        hist = np.zeros(12)
        for degree, w in weights.items():
            hist[degree] = w
        hist /= hist.sum()
        maj = _best_rotation_correlation(hist, _MAJOR_PROFILE)
        mino = _best_rotation_correlation(hist, _MINOR_PROFILE)
        wt = _best_rotation_correlation(hist, _WHOLETONE_PROFILE)
        print(f"{name:<18}{maj:>10.3f}{mino:>10.3f}{wt:>14.3f}")

    # Sanity check 3: coarse (12-bin, hard-rounded) vs fine (60-bin, smoothed
    # templates) major_fit on the SAME melody, both perfectly in-tune and
    # detuned by a random +/-40 cents per note (speech never lands exactly on
    # a 12-TET semitone). The fine version should degrade less under detuning.
    print()
    print(f"{'condition':<24}{'major_fit (12-bin)':>20}{'major_fit_fine (60-bin)':>26}")
    rng = np.random.default_rng(0)
    major_degrees_weighted = {0: 3, 4: 2, 7: 2, 2: 1, 5: 1, 9: 1, 11: 1}
    base_freq = 220.0  # A3, arbitrary tonic register

    for label, detune_cents in [("in-tune", 0.0), ("detuned +/-40 cents", 40.0)]:
        f0_events = []
        for degree, weight in major_degrees_weighted.items():
            offset_cents = rng.uniform(-detune_cents, detune_cents) if detune_cents else 0.0
            freq = base_freq * 2 ** ((degree + offset_cents / 100) / 12)
            f0_events.extend([freq] * (weight * 20))  # repeat to weight by "duration"
        f0_track = np.array(f0_events)

        coarse_hist = pitch_class_histogram(f0_track)
        fine_hist = pitch_class_histogram_fine(f0_track, n_bins=60, bandwidth=0.5)
        fine_major_template = _smooth_template(_MAJOR_PROFILE, 60, bandwidth=0.5)

        coarse_fit = _best_rotation_correlation(coarse_hist, _MAJOR_PROFILE)
        fine_fit = _best_rotation_correlation_fine(fine_hist, fine_major_template)
        print(f"{label:<24}{coarse_fit:>20.3f}{fine_fit:>26.3f}")
