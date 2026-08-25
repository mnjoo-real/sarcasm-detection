#!/usr/bin/env python
"""Final acoustic + V/A/D model (v1): 20 harmony/melody/rhythm features plus
3 wav2vec2-derived audio V/A/D features (data/audio_vad.csv, from
extract_audio_vad.py), show-normalized, evaluated with Leave-One-Show-Out
(LOSO) across all 4 MUStARD shows.

Reports weighted F1 per held-out show and the balanced average over
BBT/FRIENDS (the only two class-balanced shows -- GoldenGirls and
Sarcasmaholics are nearly single-class and inflate F1 trivially, see README).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HARMONIC_PATH = "data/harmonic_features.csv"
AUDIO_VAD_PATH = "data/audio_vad.csv"

FEATURE_COLS = [
    # harmony / prosody (13)
    "f0_mean", "f0_std", "f0_range", "jitter_local", "shimmer_local", "hnr_mean",
    "voiced_ratio", "speaking_rate_proxy", "energy_std", "dissonance_mean",
    "dissonance_std", "inharmonicity_mean", "inharmonicity_std",
    # melody (4)
    "melodic_path_length", "direction_change_rate", "final_initial_diff", "num_notes_per_sec",
    # rhythm (3)
    "stress_contrast", "ioi_cv", "rpvi_pause_filtered",
    # audio V/A/D, wav2vec2 (3)
    "audio_arousal_wav2vec", "audio_dominance_wav2vec", "audio_valence_wav2vec",
]


def normalize_by_show(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = df[c].fillna(df[c].mean())
        group_mean = df.groupby("show")[c].transform("mean")
        group_std = df.groupby("show")[c].transform("std").fillna(1).replace(0, 1)
        df[c] = (df[c] - group_mean) / group_std
    return df


def make_clf() -> SVC:
    return make_pipeline(StandardScaler(), SVC(C=10.0, gamma="scale", kernel="rbf"))


def loso_eval(df: pd.DataFrame, cols: list) -> None:
    df = normalize_by_show(df, cols)
    X, y, shows = df[cols].values, df["sarcasm"].values, df["show"].values

    f1s = {}
    for show in sorted(df["show"].unique()):
        is_test = shows == show
        train_idx, test_idx = ~is_test, is_test
        clf = make_clf().fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        report = classification_report(y[test_idx], pred, output_dict=True, zero_division=0)
        f1s[show] = report["weighted avg"]["f1-score"]
        print(f"  held-out={show:<14} n={is_test.sum():>3}  weighted F1={f1s[show]:.3f}")

    balanced = np.mean([f1s["BBT"], f1s["FRIENDS"]])
    print(f"\nBalanced (BBT+FRIENDS) show-independent F1 = {balanced:.3f}")
    print(f"All-show mean F1 = {np.mean(list(f1s.values())):.3f} "
          "(GoldenGirls/Sarcasmaholics are near single-class -- inflated, see README)")


def main() -> None:
    features = pd.read_csv(HARMONIC_PATH)
    audio_vad = pd.read_csv(AUDIO_VAD_PATH)
    df = features.merge(audio_vad, on="id", how="inner")
    print(f"n={len(df)}, n_features={len(FEATURE_COLS)}")
    loso_eval(df, FEATURE_COLS)


if __name__ == "__main__":
    main()
