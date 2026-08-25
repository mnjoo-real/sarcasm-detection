#!/usr/bin/env python
"""Trains a classifier from scratch on PodSarc's own harmony/melody/rhythm
features (not transferred from MUStARD), evaluated with GroupKFold grouped
by episode -- PodSarc's episodes share mostly the same hosts throughout, so
this tests generalization to unseen episode content/topics rather than
unseen speakers (unlike MUStARD's show-independent split).

Features are episode-normalized (z-scored within each episode) the same way
MUStARD features were show-normalized, in case different episodes have
different recording/mixing baselines.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MUSTARD_20 = [
    "f0_mean", "f0_std", "f0_range", "jitter_local", "shimmer_local", "hnr_mean",
    "voiced_ratio", "speaking_rate_proxy", "energy_std", "dissonance_mean",
    "dissonance_std", "inharmonicity_mean", "inharmonicity_std",
    "melodic_path_length", "direction_change_rate", "final_initial_diff", "num_notes_per_sec",
    "stress_contrast", "ioi_cv", "rpvi_pause_filtered",
]
FEATURE_COLS = MUSTARD_20

# Features that pass pooled + episode-controlled ANCOVA *within PodSarc itself*
# (see analyze_podsarc_replication.py) -- a from-scratch re-validation rather
# than reusing MUStARD's feature selection.
PODSARC_VALIDATED_15 = [
    "f0_mean", "voiced_ratio", "speaking_rate_proxy", "energy_std", "dissonance_mean",
    "inharmonicity_mean", "wholetone_fit", "tonal_clarity", "direction_change_rate",
    "rpvi_pause", "rpvi_pause_filtered", "stress_contrast", "ioi_cv", "ioi_npvi",
    "pitch_stress_contrast",
]
ALL_33 = None  # filled in main() from the CSV's own columns


def normalize_by_episode(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = df[c].fillna(df[c].mean())
        group_mean = df.groupby("episode")[c].transform("mean")
        group_std = df.groupby("episode")[c].transform("std").fillna(1).replace(0, 1)
        df[c] = (df[c] - group_mean) / group_std
    return df


def make_clf() -> SVC:
    return make_pipeline(StandardScaler(), SVC(C=10.0, gamma="scale", kernel="rbf"))


def evaluate(df: pd.DataFrame, cols: list, label: str, n_splits: int = 5) -> None:
    X = df[cols].values
    y = df["sarcasm"].values
    groups = df["episode"].values

    gkf = GroupKFold(n_splits=n_splits)
    f1s = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        clf = make_clf().fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        report = classification_report(y[test_idx], pred, output_dict=True, zero_division=0)
        f1s.append(report["weighted avg"]["f1-score"])
    print(f"{label:<32} n_features={len(cols):>2}  GroupKFold(episode) weighted F1 = "
          f"{np.mean(f1s):.3f} (+/- {np.std(f1s):.3f})")


def main() -> None:
    harmonic = pd.read_csv("data/podsarc_harmonic_features.csv")
    labels = pd.read_csv("data/podsarc_labels.csv")[["nid", "episode", "sarcasm"]]
    df = harmonic.merge(labels, on="nid", how="inner")
    print(f"n={len(df)}, episodes={df['episode'].nunique()}")
    all_33 = [c for c in harmonic.columns if c != "nid"]

    raw = df.copy()
    for c in all_33:
        raw[c] = raw[c].fillna(raw[c].mean())
    normed = normalize_by_episode(df, all_33)

    evaluate(raw, MUSTARD_20, "MUStARD's 20 (raw)")
    evaluate(normed, MUSTARD_20, "MUStARD's 20 (episode-normalized)")
    evaluate(normed, PODSARC_VALIDATED_15, "PodSarc-validated 15 (episode-normalized)")
    evaluate(normed, all_33, "all 33 (episode-normalized)")


if __name__ == "__main__":
    main()
