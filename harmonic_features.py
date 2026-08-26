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


# --- Rhythm pattern features --------------------------------------------------
# Rhythm = how note/rest *durations* combine into a regular or irregular
# pattern over time (long-short), plus how *strong/weak* those beats are
# (accent). Distinct from speaking_rate_proxy/energy_std above, which measure
# overall tempo and loudness variability but not the pattern of contrast
# between consecutive intervals or beats.

def _interval_durations(f0_track: np.ndarray, hop_s: float):
    """Durations (s) of contiguous voiced and unvoiced runs, in temporal order.

    A speech-rhythm-literature proxy for vocalic vs. intervocalic intervals
    (used the same way in PVI studies of speech rhythm, e.g. Grabe & Low 2002).
    """
    voiced = f0_track > 0
    if len(voiced) == 0:
        return [], []
    durations = {True: [], False: []}
    current, length = voiced[0], 1
    for v in voiced[1:]:
        if v == current:
            length += 1
        else:
            durations[current].append(length * hop_s)
            current, length = v, 1
    durations[current].append(length * hop_s)
    return durations[True], durations[False]


def _npvi(durations) -> float:
    """Normalized Pairwise Variability Index (Grabe & Low 2002): how much
    consecutive interval durations differ, relative to their size. High =
    contrastive/irregular long-short pattern; low = evenly-timed."""
    if len(durations) < 2:
        return np.nan
    d = np.array(durations)
    d1, d2 = d[:-1], d[1:]
    denom = (d1 + d2) / 2
    valid = denom > 0
    if not np.any(valid):
        return np.nan
    return float(100 * np.mean(np.abs(d1[valid] - d2[valid]) / denom[valid]))


def _rpvi(durations) -> float:
    """Raw Pairwise Variability Index (unnormalized, standard for consonantal/
    pause intervals in the speech-rhythm literature)."""
    if len(durations) < 2:
        return np.nan
    d = np.array(durations)
    return float(np.mean(np.abs(d[:-1] - d[1:])))


def rhythm_pattern_features(sound: parselmouth.Sound, hop_s: float = 0.02) -> dict:
    pitch = sound.to_pitch(time_step=hop_s)
    f0_track = pitch.selected_array["frequency"]
    voiced_durs, unvoiced_durs = _interval_durations(f0_track, hop_s)

    intensity = sound.to_intensity()
    values = intensity.values[0]
    values = values[np.isfinite(values)]
    time_step = intensity.time_step

    stress_contrast, ioi_cv, ioi_npvi = np.nan, np.nan, np.nan
    if len(values) > 0:
        threshold = np.mean(values) - 3
        min_distance = max(int(0.08 / time_step), 1)
        peaks, _ = find_peaks(values, height=threshold, distance=min_distance)

        if len(peaks) > 1:
            # How much consecutive beats alternate strong/weak (not peak-vs-
            # silence loudness, which is a different, unrelated quantity).
            peak_heights = values[peaks]
            stress_contrast = float(np.mean(np.abs(np.diff(peak_heights))))

        if len(peaks) > 2:
            iois = np.diff(peaks) * time_step
            if np.mean(iois) > 0:
                ioi_cv = float(np.std(iois) / np.mean(iois))
            # ioi_cv is a *global* variability measure, so a single outlier
            # (e.g. utterance-final lengthening, unrelated to sarcasm) can
            # dominate it. ioi_npvi only compares each IOI to its immediate
            # neighbor, so one boundary outlier affects just one pair instead
            # of the whole statistic.
            ioi_npvi = _npvi(iois)

    min_pause_s = 0.10  # below this is almost certainly a consonant closure, not a real pause
    real_pauses = [d for d in unvoiced_durs if d >= min_pause_s]

    return {
        "npvi_voiced": _npvi(voiced_durs),
        "rpvi_pause": _rpvi(unvoiced_durs),
        "rpvi_pause_filtered": _rpvi(real_pauses),
        "stress_contrast": stress_contrast,
        "ioi_cv": ioi_cv,
        "ioi_npvi": ioi_npvi,
    }


def pitch_stress_contrast_features(sound: parselmouth.Sound, hop_s: float = 0.02,
                                    ref_freq: float = 440.0) -> dict:
    """Pitch-accent analogue of stress_contrast: how much F0 (in semitones)
    differs between consecutive syllable-nucleus peaks, rather than loudness.

    Loudness-based stress_contrast can be confounded by per-recording gain/mic
    differences even after show-normalization; pitch accent is a second,
    largely independent marker of prominence in speech and isn't tied to
    absolute recording level.
    """
    intensity = sound.to_intensity()
    values = intensity.values[0]
    times = intensity.ts()
    valid = np.isfinite(values)
    values, times = values[valid], times[valid]

    if len(values) == 0:
        return {"pitch_stress_contrast": np.nan}

    threshold = np.mean(values) - 3
    min_distance = max(int(0.08 / intensity.time_step), 1)
    peaks, _ = find_peaks(values, height=threshold, distance=min_distance)

    if len(peaks) < 2:
        return {"pitch_stress_contrast": np.nan}

    pitch = sound.to_pitch(time_step=hop_s)
    peak_semitones = []
    for idx in peaks:
        f0 = pitch.get_value_at_time(times[idx])
        if f0 is not None and not np.isnan(f0) and f0 > 0:
            peak_semitones.append(12 * np.log2(f0 / ref_freq))

    if len(peak_semitones) < 2:
        return {"pitch_stress_contrast": np.nan}

    return {"pitch_stress_contrast": float(np.mean(np.abs(np.diff(peak_semitones))))}


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


# --- Melody features ---------------------------------------------------------
# Melody = pitch movement organized in time (with rhythm), as distinct from
# harmony (simultaneous pitch relationships, covered above) and from the
# static F0 mean/std/range already in prosody_features. Two complementary
# views: (A) continuous contour shape/activity, (B) a discretized "note"
# sequence (via stable-region segmentation) and its interval structure.

def _voiced_runs(f0_track: np.ndarray):
    """Yields (start, end) index pairs for each contiguous run of voiced frames."""
    voiced = f0_track > 0
    runs = []
    start = None
    for i, v in enumerate(voiced):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(voiced)))
    return runs


def melody_contour_features(sound: parselmouth.Sound, hop_s: float = 0.02,
                             ref_freq: float = 440.0, min_step_semitone: float = 0.1) -> dict:
    """Continuous-contour melody features: how much and which way the pitch moves.

    min_step_semitone filters out frame-to-frame jitter: Praat's pitch tracker
    introduces sub-0.1-semitone noise even on a perfectly held tone, and without
    a floor, direction_change_rate ends up counting jitter reversals rather than
    real melodic turns (confirmed via the synthetic 4-note sanity check below).
    """
    pitch = sound.to_pitch(time_step=hop_s)
    f0_track = pitch.selected_array["frequency"]
    times = pitch.ts()
    duration = sound.get_total_duration()

    total_path = 0.0
    direction_changes = 0
    all_t, all_s = [], []

    for start, end in _voiced_runs(f0_track):
        if end - start < 2:
            continue
        seg_semitone = 12 * np.log2(f0_track[start:end] / ref_freq)
        seg_t = times[start:end]
        diffs = np.diff(seg_semitone)
        total_path += float(np.sum(np.abs(diffs)))

        significant = diffs[np.abs(diffs) > min_step_semitone]
        signs = np.sign(significant)
        if len(signs) > 1:
            direction_changes += int(np.sum(signs[1:] != signs[:-1]))

        all_t.extend(seg_t)
        all_s.extend(seg_semitone)

    if len(all_t) < 2:
        return {"melodic_path_length": np.nan, "direction_change_rate": np.nan,
                "contour_slope": np.nan, "final_initial_diff": np.nan}

    all_t, all_s = np.array(all_t), np.array(all_s)
    slope = float(np.polyfit(all_t, all_s, 1)[0])

    return {
        "melodic_path_length": total_path / duration if duration > 0 else np.nan,
        "direction_change_rate": direction_changes / duration if duration > 0 else np.nan,
        "contour_slope": slope,
        "final_initial_diff": float(all_s[-1] - all_s[0]),
    }


def segment_notes(f0_track: np.ndarray, times: np.ndarray, ref_freq: float = 440.0,
                   velocity_thresh: float = 8.0, min_note_duration: float = 0.05,
                   hop_s: float = 0.02):
    """Splits a continuous F0 track into discrete "notes": stable-pitch regions
    where the glide velocity stays below velocity_thresh (semitones/sec).

    Returns a list of (pitch_semitone, duration_s) tuples, one per detected note.
    """
    notes = []
    for start, end in _voiced_runs(f0_track):
        if end - start < 3:
            continue
        seg_semitone = 12 * np.log2(f0_track[start:end] / ref_freq)
        seg_t = times[start:end]
        velocity = np.gradient(seg_semitone, seg_t)
        stable = np.abs(velocity) < velocity_thresh

        idx, n = 0, len(stable)
        while idx < n:
            if stable[idx]:
                j = idx
                while j < n and stable[j]:
                    j += 1
                note_duration = seg_t[j - 1] - seg_t[idx] + hop_s
                if note_duration >= min_note_duration:
                    notes.append((float(np.median(seg_semitone[idx:j])), float(note_duration)))
                idx = j
            else:
                idx += 1
    return notes


def melodic_interval_features(sound: parselmouth.Sound, hop_s: float = 0.02,
                               ref_freq: float = 440.0) -> dict:
    """Discretizes the pitch contour into notes, then measures the melodic
    interval structure between consecutive notes (a speech analogue of a
    melodic-interval-content analysis in music)."""
    pitch = sound.to_pitch(time_step=hop_s)
    f0_track = pitch.selected_array["frequency"]
    times = pitch.ts()
    duration = sound.get_total_duration()

    notes = segment_notes(f0_track, times, ref_freq=ref_freq, hop_s=hop_s)

    if len(notes) < 2:
        return {"mean_interval_size": np.nan, "interval_std": np.nan,
                "num_notes_per_sec": np.nan, "direction_entropy": np.nan}

    pitches = np.array([note[0] for note in notes])
    intervals = np.diff(pitches)
    directions = np.sign(intervals)

    _, counts = np.unique(directions, return_counts=True)
    probs = counts / counts.sum()
    direction_entropy = float(-np.sum(probs * np.log2(probs)))

    return {
        "mean_interval_size": float(np.mean(np.abs(intervals))),
        "interval_std": float(np.std(intervals)),
        "num_notes_per_sec": float(len(notes) / duration) if duration > 0 else np.nan,
        "direction_entropy": direction_entropy,
    }


# --- Formants (articulation) -------------------------------------------------
# Vocal-tract resonances (F1/F2/F3) -- an articulation axis this project
# hasn't touched at all: pitch (F0), timbre (dissonance/MFCC), and rhythm are
# already covered, but not "how exaggerated or constrained is the mouth
# shape." A wider or more variable formant spread could mark exaggerated
# articulation used for comic delivery.

def formant_features(sound: parselmouth.Sound, hop_s: float = 0.01) -> dict:
    formant = sound.to_formant_burg()
    duration = sound.get_total_duration()
    times = np.arange(0, duration, hop_s)

    f1_vals, f2_vals, f3_vals = [], [], []
    for t in times:
        f1 = formant.get_value_at_time(1, t)
        f2 = formant.get_value_at_time(2, t)
        f3 = formant.get_value_at_time(3, t)
        if f1 is not None and not np.isnan(f1):
            f1_vals.append(f1)
        if f2 is not None and not np.isnan(f2):
            f2_vals.append(f2)
        if f3 is not None and not np.isnan(f3):
            f3_vals.append(f3)

    def stats(vals, name):
        if not vals:
            return {f"{name}_mean": np.nan, f"{name}_std": np.nan}
        return {f"{name}_mean": float(np.mean(vals)), f"{name}_std": float(np.std(vals))}

    result = {}
    result.update(stats(f1_vals, "f1"))
    result.update(stats(f2_vals, "f2"))
    result.update(stats(f3_vals, "f3"))

    # Formant dispersion: average adjacent-formant spacing, a standard proxy
    # for vocal-tract length / how open the articulatory setting is.
    n = min(len(f1_vals), len(f2_vals), len(f3_vals))
    if n > 0:
        f1a, f2a, f3a = np.array(f1_vals[:n]), np.array(f2_vals[:n]), np.array(f3_vals[:n])
        result["formant_dispersion"] = float(np.mean([(f2a - f1a).mean(), (f3a - f2a).mean()]))
    else:
        result["formant_dispersion"] = np.nan
    return result


# --- Voice quality: CPP and spectral tilt (H1-H2) ----------------------------
# More specific voice-quality measures than HNR: Cepstral Peak Prominence
# (CPP) is widely considered the more robust clinical/research standard for
# periodicity strength, and H1-H2 (spectral tilt between the first two
# harmonics) directly targets breathy-vs-pressed vocal effort, rather than
# HNR's coarser harmonic-vs-noise split.

def _cepstral_peak_prominence(frame: np.ndarray, sr: int, f0_min: float = 60.0,
                               f0_max: float = 400.0) -> float:
    windowed = frame * np.hanning(len(frame))
    spectrum = np.abs(np.fft.rfft(windowed))
    log_spectrum = np.log(spectrum + 1e-10)
    cepstrum = np.fft.irfft(log_spectrum)

    quefrency = np.arange(len(cepstrum)) / sr
    min_q, max_q = 1.0 / f0_max, 1.0 / f0_min
    valid = (quefrency >= min_q) & (quefrency <= max_q)
    if not np.any(valid):
        return np.nan

    valid_idx = np.where(valid)[0]
    peak_idx = valid_idx[np.argmax(cepstrum[valid_idx])]
    peak_val = cepstrum[peak_idx]

    fit_range = (quefrency > 1e-3) & (quefrency < max_q * 1.5)
    if np.sum(fit_range) < 2:
        return np.nan
    coeffs = np.polyfit(quefrency[fit_range], cepstrum[fit_range], 1)
    trend_at_peak = np.polyval(coeffs, quefrency[peak_idx])
    return float(peak_val - trend_at_peak)


def _h1_h2(frame: np.ndarray, sr: int, f0: float, tol_hz: float = 20.0) -> float:
    if f0 <= 0:
        return np.nan
    windowed = frame * np.hanning(len(frame))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(frame), d=1.0 / sr)

    def amp_near(target_freq):
        idx = np.argmin(np.abs(freqs - target_freq))
        return spectrum[idx]

    h1, h2 = amp_near(f0), amp_near(2 * f0)
    if h1 <= 0 or h2 <= 0:
        return np.nan
    return float(20 * np.log10(h1) - 20 * np.log10(h2))


def voice_quality_features(sound: parselmouth.Sound, frame_length_s: float = 0.04,
                            hop_s: float = 0.02) -> dict:
    y = sound.values[0]
    sr = int(sound.sampling_frequency)

    pitch = sound.to_pitch(time_step=hop_s)
    f0_track = pitch.selected_array["frequency"]
    times = pitch.ts()

    frame_len = int(frame_length_s * sr)
    cpps, h1h2s = [], []

    for t, f0 in zip(times, f0_track):
        if f0 <= 0:
            continue
        center = int(t * sr)
        start, end = center - frame_len // 2, center + frame_len // 2
        if start < 0 or end > len(y):
            continue
        frame = y[start:end]
        cpps.append(_cepstral_peak_prominence(frame, sr))
        h1h2s.append(_h1_h2(frame, sr, f0))

    def stats(values, name):
        values = [v for v in values if not np.isnan(v)]
        if not values:
            return {f"{name}_mean": np.nan, f"{name}_std": np.nan}
        return {f"{name}_mean": float(np.mean(values)), f"{name}_std": float(np.std(values))}

    result = {}
    result.update(stats(cpps, "cpp"))
    result.update(stats(h1h2s, "h1h2"))
    return result


def extract_all_features(path: str) -> dict:
    sound = parselmouth.Sound(path)
    if sound.get_number_of_channels() > 1:
        sound = sound.convert_to_mono()
    features = {}
    features.update(prosody_features(sound))
    features.update(rhythm_features(sound))
    features.update(harmony_features(sound))
    features.update(scale_features(sound))
    features.update(melody_contour_features(sound))
    features.update(melodic_interval_features(sound))
    features.update(rhythm_pattern_features(sound))
    features.update(pitch_stress_contrast_features(sound))
    features.update(formant_features(sound))
    features.update(voice_quality_features(sound))
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

    # Sanity check for melody features: synthesize a clean 4-note melody
    # (A3 -> B3 -> C#4 -> A3, i.e. intervals +2, +2, -4 semitones) with short
    # glides between held notes, and confirm segment_notes recovers ~4 notes
    # with intervals matching what was actually played.
    print()
    print("=== Melody sanity check: synthetic A3-B3-C#4-A3 melody ===")
    sr = 16000
    note_freqs = [220.00, 246.94, 277.18, 220.00]  # A3, B3, C#4, A3
    note_dur, glide_dur = 0.3, 0.03
    segments = []
    for i, f in enumerate(note_freqs):
        segments.append(np.full(int(note_dur * sr), f))
        if i < len(note_freqs) - 1:
            segments.append(np.linspace(f, note_freqs[i + 1], int(glide_dur * sr)))
    freq_track = np.concatenate(segments)
    t = np.arange(len(freq_track)) / sr
    phase = 2 * np.pi * np.cumsum(freq_track) / sr
    y = np.sin(phase)
    sf_path = "/private/tmp/claude-501/-Users-minjoolee-Downloads-MUStARD/c8ce0015-f13f-4958-aeaa-1445bff3da93/scratchpad/_melody_sanity_check.wav"
    try:
        import soundfile as sf
        sf.write(sf_path, y, sr)
        sound = parselmouth.Sound(sf_path)
        contour = melody_contour_features(sound)
        interval = melodic_interval_features(sound)
        print("expected intervals: [+2, +2, -4] semitones, ~4 notes, ~3.1 notes/sec")
        print({**contour, **interval})
    except Exception as e:
        print(f"skipped (needs soundfile + a writable /tmp): {e}")

    # Sanity check: coarse (12-bin, hard-rounded) vs fine (60-bin, smoothed
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

    # Sanity check for rhythm features: a perfectly regular beat (equal note/
    # rest durations, equal loudness) should score low on nPVI/rPVI/ioi_cv/
    # stress_contrast; an irregular, accented beat (alternating long/short
    # notes, alternating loud/soft) should score high on all four.
    print()
    print("=== Rhythm sanity check: regular vs irregular+accented beat ===")

    def make_beat_track(sr, durations, amps, gap, freqs=None, fade_s=0.01):
        if freqs is None:
            freqs = [220.0] * len(durations)
        segs = []
        for dur, amp, freq in zip(durations, amps, freqs):
            n = int(dur * sr)
            tone = amp * np.sin(2 * np.pi * freq * np.arange(n) / sr)
            fade_n = min(int(fade_s * sr), n // 2)
            if fade_n > 0:
                ramp = np.linspace(0, 1, fade_n)
                tone[:fade_n] *= ramp
                tone[-fade_n:] *= ramp[::-1]
            segs.append(tone)
            segs.append(np.zeros(int(gap * sr)))
        return np.concatenate(segs)

    sr = 16000
    regular = make_beat_track(sr, durations=[0.2] * 8, amps=[0.8] * 8, gap=0.2)
    irregular = make_beat_track(sr, durations=[0.15, 0.35] * 4, amps=[0.9, 0.3] * 4, gap=0.2)

    print(f"{'condition':<14}{'nPVI_voiced':>12}{'rPVI_pause':>12}{'stress_contrast':>17}{'ioi_cv':>10}")
    for name, y in [("regular", regular), ("irregular", irregular)]:
        path = f"/private/tmp/claude-501/-Users-minjoolee-Downloads-MUStARD/c8ce0015-f13f-4958-aeaa-1445bff3da93/scratchpad/_rhythm_sanity_{name}.wav"
        try:
            import soundfile as sf
            sf.write(path, y, sr)
            sound = parselmouth.Sound(path)
            feats = rhythm_pattern_features(sound)
            print(f"{name:<14}{feats['npvi_voiced']:>12.3f}{feats['rpvi_pause']:>12.4f}"
                  f"{feats['stress_contrast']:>17.3f}{feats['ioi_cv']:>10.3f}")
        except Exception as e:
            print(f"{name}: skipped ({e})")

    # Sanity check for pitch_stress_contrast: same constant loudness/duration
    # throughout (so the loudness-based stress_contrast should stay ~0 in both
    # cases), but one version alternates pitch (A3/E4, a 7-semitone leap) and
    # the other stays flat. pitch_stress_contrast should track ONLY the pitch
    # alternation, confirming it captures something stress_contrast can't.
    print()
    print("=== Pitch-accent sanity check: flat pitch vs alternating pitch, constant loudness ===")
    flat = make_beat_track(sr, durations=[0.2] * 8, amps=[0.8] * 8, gap=0.2,
                            freqs=[220.0] * 8)
    alternating = make_beat_track(sr, durations=[0.2] * 8, amps=[0.8] * 8, gap=0.2,
                                   freqs=[220.0, 329.63] * 4)  # A3, E4 (perfect 5th, 7 semitones)

    print(f"{'condition':<14}{'stress_contrast':>17}{'pitch_stress_contrast':>24}")
    for name, y in [("flat pitch", flat), ("alternating pitch", alternating)]:
        path = f"/private/tmp/claude-501/-Users-minjoolee-Downloads-MUStARD/c8ce0015-f13f-4958-aeaa-1445bff3da93/scratchpad/_pitch_stress_{name.replace(' ', '_')}.wav"
        try:
            import soundfile as sf
            sf.write(path, y, sr)
            sound = parselmouth.Sound(path)
            loud_feats = rhythm_pattern_features(sound)
            pitch_feats = pitch_stress_contrast_features(sound)
            print(f"{name:<14}{loud_feats['stress_contrast']:>17.3f}"
                  f"{pitch_feats['pitch_stress_contrast']:>24.3f}")
        except Exception as e:
            print(f"{name}: skipped ({e})")

    # Sanity check: ioi_npvi vs ioi_cv robustness to a single boundary outlier
    # (e.g. utterance-final lengthening -- a common, sarcasm-unrelated effect).
    # A local pairwise measure (npvi) should move much less than a global one
    # (cv) when only the very last interval is stretched.
    print()
    print("=== ioi_cv vs ioi_npvi: robustness to one outlier IOI ===")
    regular_iois = np.array([0.4] * 7)
    with_outlier = np.array([0.4] * 6 + [1.2])  # last IOI stretched (final lengthening)

    for label, iois in [("no outlier", regular_iois), ("with 1 outlier", with_outlier)]:
        cv = float(np.std(iois) / np.mean(iois))
        npvi = _npvi(iois)
        print(f"{label:<18} ioi_cv={cv:.3f}  ioi_npvi={npvi:.3f}")

    # Sanity check: rpvi_pause_filtered should ignore short consonant-closure
    # gaps and reflect only the variability among genuine (>=100ms) pauses.
    print()
    print("=== rpvi_pause vs rpvi_pause_filtered: consonant closures mixed with real pauses ===")
    consonant_gaps = [0.03, 0.04, 0.035, 0.045, 0.03]
    real_pauses = [0.30, 0.45, 0.32]
    mixed = consonant_gaps + real_pauses
    filtered = [d for d in mixed if d >= 0.10]
    print(f"raw rpvi_pause (mixed)      = {_rpvi(mixed):.4f}")
    print(f"filtered rpvi_pause (>=100ms) = {_rpvi(filtered):.4f}  (expected: reflects only {real_pauses})")

    # Sanity check: formants. Synthesize a vowel via a cascade of resonant
    # filters (source-filter model) at known formant frequencies and confirm
    # formant_features recovers values close to what was actually built in.
    print()
    print("=== Formant sanity check: synthetic vowel with known F1/F2/F3 ===")
    import scipy.signal as sig
    import soundfile as sf

    def synth_vowel(f0, formants, bandwidths, duration, sr):
        n = int(duration * sr)
        source = np.zeros(n)
        period = max(int(sr / f0), 1)
        source[::period] = 1.0
        y = source.copy()
        for f, bw in zip(formants, bandwidths):
            r = np.exp(-np.pi * bw / sr)
            theta = 2 * np.pi * f / sr
            a1, a2 = 2 * r * np.cos(theta), -r ** 2
            y = sig.lfilter([1 - a1 - a2], [1, -a1, -a2], y)
        return y / (np.max(np.abs(y)) + 1e-9) * 0.8

    sr = 16000
    target_formants = [700, 1220, 2600]  # approx an "ah"-like vowel
    y = synth_vowel(f0=120, formants=target_formants, bandwidths=[80, 90, 120], duration=0.6, sr=sr)
    path = "/private/tmp/claude-501/-Users-minjoolee-Downloads-MUStARD/c8ce0015-f13f-4958-aeaa-1445bff3da93/scratchpad/_formant_sanity.wav"
    sf.write(path, y, sr)
    sound = parselmouth.Sound(path)
    f_feats = formant_features(sound)
    print(f"target F1/F2/F3 = {target_formants}")
    print(f"recovered: F1={f_feats['f1_mean']:.0f}  F2={f_feats['f2_mean']:.0f}  F3={f_feats['f3_mean']:.0f}")

    # Sanity check: CPP should be high for a clean periodic tone and near-zero
    # (no prominent cepstral peak) for white noise.
    print()
    print("=== CPP sanity check: periodic tone vs white noise ===")
    t = np.arange(int(0.05 * sr)) / sr
    f0_test = 150
    periodic_frame = sum(np.sin(2 * np.pi * h * f0_test * t) / h for h in range(1, 15))
    rng = np.random.default_rng(0)
    noise_frame = rng.normal(0, 1, len(t))
    print(f"CPP (periodic tone) = {_cepstral_peak_prominence(periodic_frame, sr):.3f}")
    print(f"CPP (white noise)   = {_cepstral_peak_prominence(noise_frame, sr):.3f}  (expect much lower)")

    # Sanity check: H1-H2 should be larger (more positive) for a breathy
    # harmonic profile (steep falloff) than a pressed one (flatter falloff).
    print()
    print("=== H1-H2 sanity check: breathy vs pressed harmonic profile ===")
    f0 = 150
    breathy = sum((1.0 / (h ** 2.5)) * np.sin(2 * np.pi * h * f0 * t) for h in range(1, 8))
    pressed = sum((1.0 / h) * np.sin(2 * np.pi * h * f0 * t) for h in range(1, 8))
    print(f"H1-H2 (breathy)  = {_h1_h2(breathy, sr, f0):.2f} dB")
    print(f"H1-H2 (pressed)  = {_h1_h2(pressed, sr, f0):.2f} dB  (expect lower/more negative than breathy)")
