#!/usr/bin/env python
"""Same as train_harmonic_svm.py, but each feature is first z-scored within its
own show (per-show mean/std subtracted) before classification.

Rationale: the raw-feature show-independent test collapsed to chance (F1=0.494,
see train_harmonic_svm.py), while individual features stayed significant even
after controlling for show in an ANCOVA. That gap suggests the SVM was fitting
each show's absolute baseline level rather than the within-show relative shift
that actually carries the sarcasm signal. Per-show normalization removes the
baseline and keeps only the relative deviation, mirroring standard speaker
normalization in paralinguistics/emotion recognition.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEATURES_PATH = "data/harmonic_features.csv"
NON_FEATURE_COLS = {"id", "sarcasm", "show", "speaker"}


def load_xy_normalized(df: pd.DataFrame, group_col: str = "speaker"):
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    df = df.copy()
    for c in feature_cols:
        df[c] = df[c].fillna(df[c].mean())
        group_mean = df.groupby(group_col)[c].transform("mean")
        group_std = df.groupby(group_col)[c].transform("std").fillna(1).replace(0, 1)
        df[c] = (df[c] - group_mean) / group_std
    X = df[feature_cols].values
    y = df["sarcasm"].values
    return X, y, feature_cols


def make_clf() -> SVC:
    return make_pipeline(StandardScaler(), SVC(C=10.0, gamma="scale", kernel="rbf"))


def random_cv(X: np.ndarray, y: np.ndarray) -> None:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    f1s = []
    for train_idx, test_idx in skf.split(X, y):
        clf = make_clf().fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        report = classification_report(y[test_idx], pred, output_dict=True, digits=3)
        f1s.append(report["weighted avg"]["f1-score"])
    print(f"Random 5-fold CV (speaker-normalized): weighted F1 = {np.mean(f1s):.3f} (+/- {np.std(f1s):.3f})")


def show_independent(df: pd.DataFrame, X: np.ndarray, y: np.ndarray) -> None:
    is_friends = (df["show"] == "FRIENDS").values
    train_idx, test_idx = ~is_friends, is_friends

    clf = make_clf().fit(X[train_idx], y[train_idx])
    pred = clf.predict(X[test_idx])

    print("Show-independent, speaker-normalized (train=BBT+GOLDENGIRLS+SARCASMOHOLICS, test=FRIENDS):")
    print(classification_report(y[test_idx], pred, digits=3))


def main() -> None:
    df = pd.read_csv(FEATURES_PATH)
    X, y, _ = load_xy_normalized(df)
    random_cv(X, y)
    print()
    show_independent(df, X, y)


if __name__ == "__main__":
    main()
