#!/usr/bin/env python
"""Out-of-corpus validation: train the final 23-feature model on ALL of
MUStARD (no holdout -- PodSarc itself is now the test set), then evaluate on
PodSarc's human-verified sarcasm subset.

Normalization: MUStARD features are z-scored per-show as usual. PodSarc has
no equivalent "show" grouping across its own data in our pipeline, so it is
normalized as a single group using its own mean/std -- the natural extension
of "normalize away this corpus's recording/register baseline" to a genuinely
new corpus, consistent with why show-normalization was needed in the first
place (baseline differences, not sarcasm, otherwise leak into the features).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEATURE_COLS = [
    "f0_mean", "f0_std", "f0_range", "jitter_local", "shimmer_local", "hnr_mean",
    "voiced_ratio", "speaking_rate_proxy", "energy_std", "dissonance_mean",
    "dissonance_std", "inharmonicity_mean", "inharmonicity_std",
    "melodic_path_length", "direction_change_rate", "final_initial_diff", "num_notes_per_sec",
    "stress_contrast", "ioi_cv", "rpvi_pause_filtered",
    "audio_arousal_wav2vec", "audio_dominance_wav2vec", "audio_valence_wav2vec",
]


def normalize_by_group(df: pd.DataFrame, cols: list, group_col: str) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = df[c].fillna(df[c].mean())
        group_mean = df.groupby(group_col)[c].transform("mean")
        group_std = df.groupby(group_col)[c].transform("std").fillna(1).replace(0, 1)
        df[c] = (df[c] - group_mean) / group_std
    return df


def main() -> None:
    mustard = pd.read_csv("data/harmonic_features.csv").merge(
        pd.read_csv("data/audio_vad.csv"), on="id", how="inner")
    mustard = normalize_by_group(mustard, FEATURE_COLS, "show")

    podsarc = pd.read_csv("data/podsarc_harmonic_features.csv").merge(
        pd.read_csv("data/podsarc_audio_vad.csv"), on="nid", how="inner")
    labels = pd.read_csv("data/podsarc_labels_sample.csv")[["nid", "sarcasm"]]
    podsarc = podsarc.merge(labels, on="nid", how="inner")
    podsarc["corpus"] = "podsarc"
    podsarc = normalize_by_group(podsarc, FEATURE_COLS, "corpus")

    print(f"MUStARD n={len(mustard)}, PodSarc n={len(podsarc)}")
    print("PodSarc label balance:\n", podsarc["sarcasm"].value_counts())

    X_train, y_train = mustard[FEATURE_COLS].values, mustard["sarcasm"].values
    X_test, y_test = podsarc[FEATURE_COLS].values, podsarc["sarcasm"].values

    clf = make_pipeline(StandardScaler(), SVC(C=10.0, gamma="scale", kernel="rbf"))
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    print("\n=== MUStARD-trained model on PodSarc (out-of-corpus) ===")
    print(classification_report(y_test, pred, digits=3))
    print("confusion matrix:\n", confusion_matrix(y_test, pred))

    # Reference point: how the same model does on a MUStARD-internal random split.
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    f1s = []
    for tr, te in skf.split(X_train, y_train):
        c2 = make_pipeline(StandardScaler(), SVC(C=10.0, gamma="scale", kernel="rbf"))
        c2.fit(X_train[tr], y_train[tr])
        p2 = c2.predict(X_train[te])
        f1s.append(classification_report(y_train[te], p2, output_dict=True)["weighted avg"]["f1-score"])
    print(f"\n(reference) MUStARD-internal random 5-fold CV weighted F1 = {np.mean(f1s):.3f}")


if __name__ == "__main__":
    main()
