#!/usr/bin/env python
"""Trains an SVM on the harmonic/prosodic features (data/harmonic_features.csv)
and evaluates it two ways:
  1. Random stratified 5-fold CV (like MUStARD's speaker-dependent setting).
  2. Show-independent hold-out: train on BBT+GOLDENGIRLS+SARCASMOHOLICS, test on
     FRIENDS -- mirrors MUStARD's own speaker_independent_split (data_loader.py),
     where FRIENDS is held out entirely so no actor's voice is seen in training.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEATURES_PATH = "data/harmonic_features.csv"
NON_FEATURE_COLS = {"id", "sarcasm", "show"}


def load_xy(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].fillna(df[feature_cols].mean()).values
    y = df["sarcasm"].values
    return X, y, feature_cols


def make_clf() -> SVC:
    return make_pipeline(StandardScaler(), SVC(C=10.0, gamma="scale", kernel="rbf"))


def random_cv(df: pd.DataFrame) -> None:
    X, y, _ = load_xy(df)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    f1s = []
    for train_idx, test_idx in skf.split(X, y):
        clf = make_clf().fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        report = classification_report(y[test_idx], pred, output_dict=True, digits=3)
        f1s.append(report["weighted avg"]["f1-score"])
    print(f"Random 5-fold CV: weighted F1 = {np.mean(f1s):.3f} (+/- {np.std(f1s):.3f})")


def show_independent(df: pd.DataFrame) -> None:
    X, y, _ = load_xy(df)
    is_friends = (df["show"] == "FRIENDS").values

    train_idx, test_idx = ~is_friends, is_friends
    clf = make_clf().fit(X[train_idx], y[train_idx])
    pred = clf.predict(X[test_idx])

    print("Show-independent (train=BBT+GOLDENGIRLS+SARCASMOHOLICS, test=FRIENDS):")
    print(classification_report(y[test_idx], pred, digits=3))


def main() -> None:
    df = pd.read_csv(FEATURES_PATH)
    random_cv(df)
    print()
    show_independent(df)


if __name__ == "__main__":
    main()
