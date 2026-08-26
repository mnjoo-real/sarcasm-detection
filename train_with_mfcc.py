#!/usr/bin/env python
"""Adds MFCC (+spectral centroid, 28-dim) to the current best feature set for
either corpus and re-checks show/episode-independent performance, to see
whether the Phase-1 MUStARD finding (MFCC hurts show-independent F1) still
holds now that the feature set has grown well past the original 13.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MUSTARD_BEST_23 = [
    "f0_mean", "f0_std", "f0_range", "jitter_local", "shimmer_local", "hnr_mean",
    "voiced_ratio", "speaking_rate_proxy", "energy_std", "dissonance_mean",
    "dissonance_std", "inharmonicity_mean", "inharmonicity_std",
    "melodic_path_length", "direction_change_rate", "final_initial_diff", "num_notes_per_sec",
    "stress_contrast", "ioi_cv", "rpvi_pause_filtered",
    "audio_arousal_wav2vec", "audio_dominance_wav2vec", "audio_valence_wav2vec",
]
PODSARC_VALIDATED_15 = [
    "f0_mean", "voiced_ratio", "speaking_rate_proxy", "energy_std", "dissonance_mean",
    "inharmonicity_mean", "wholetone_fit", "tonal_clarity", "direction_change_rate",
    "rpvi_pause", "rpvi_pause_filtered", "stress_contrast", "ioi_cv", "ioi_npvi",
    "pitch_stress_contrast",
]
MFCC_COLS = [f"mfcc{i}_{stat}" for i in range(1, 14) for stat in ("mean", "std")] + [
    "spectral_centroid_mean", "spectral_centroid_std"]


def normalize_by_group(df: pd.DataFrame, cols: list, group_col: str) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = df[c].fillna(df[c].mean())
        gm = df.groupby(group_col)[c].transform("mean")
        gs = df.groupby(group_col)[c].transform("std").fillna(1).replace(0, 1)
        df[c] = (df[c] - gm) / gs
    return df


def make_clf() -> SVC:
    return make_pipeline(StandardScaler(), SVC(C=10.0, gamma="scale", kernel="rbf"))


def loso_mustard(df: pd.DataFrame, cols: list, label: str) -> None:
    dfn = normalize_by_group(df, cols, "show")
    X, y, shows = dfn[cols].values, dfn["sarcasm"].values, dfn["show"].values
    f1s = {}
    for show in sorted(dfn["show"].unique()):
        is_test = shows == show
        clf = make_clf().fit(X[~is_test], y[~is_test])
        pred = clf.predict(X[is_test])
        rep = classification_report(y[is_test], pred, output_dict=True, zero_division=0)
        f1s[show] = rep["weighted avg"]["f1-score"]
    balanced = np.mean([f1s["BBT"], f1s["FRIENDS"]])
    print(f"{label:<32} n={len(cols):>2}  BBT={f1s['BBT']:.3f}  FRIENDS={f1s['FRIENDS']:.3f}  balanced={balanced:.3f}")


def groupkfold_podsarc(df: pd.DataFrame, cols: list, label: str, n_splits: int = 5) -> None:
    dfn = normalize_by_group(df, cols, "episode")
    X, y, groups = dfn[cols].values, dfn["sarcasm"].values, dfn["episode"].values
    gkf = GroupKFold(n_splits=n_splits)
    f1s = []
    for tr, te in gkf.split(X, y, groups):
        clf = make_clf().fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        rep = classification_report(y[te], pred, output_dict=True, zero_division=0)
        f1s.append(rep["weighted avg"]["f1-score"])
    print(f"{label:<32} n={len(cols):>2}  F1={np.mean(f1s):.3f} (+/- {np.std(f1s):.3f})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=["mustard", "podsarc"], required=True)
    args = parser.parse_args()

    if args.corpus == "mustard":
        features = pd.read_csv("data/harmonic_features.csv").merge(
            pd.read_csv("data/audio_vad.csv"), on="id", how="inner")
        mfcc = pd.read_csv("data/mfcc_features.csv")
        df = features.merge(mfcc, on="id", how="inner")
        loso_mustard(df, MUSTARD_BEST_23, "best 23 (no MFCC)")
        loso_mustard(df, MUSTARD_BEST_23 + MFCC_COLS, "best 23 + MFCC (51)")
    else:
        harmonic = pd.read_csv("data/podsarc_harmonic_features.csv")
        labels = pd.read_csv("data/podsarc_labels.csv")[["nid", "episode", "sarcasm"]]
        mfcc = pd.read_csv("data/podsarc_mfcc_features.csv").rename(columns={"id": "nid"})
        df = harmonic.merge(labels, on="nid").merge(mfcc, on="nid")
        groupkfold_podsarc(df, PODSARC_VALIDATED_15, "PodSarc-validated 15 (no MFCC)")
        groupkfold_podsarc(df, PODSARC_VALIDATED_15 + MFCC_COLS, "PodSarc-validated 15 + MFCC (43)")


if __name__ == "__main__":
    main()
