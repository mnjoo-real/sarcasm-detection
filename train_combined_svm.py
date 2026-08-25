#!/usr/bin/env python
"""Combines harmonic/prosodic features with MFCC features, applies per-show
normalization (the best-performing setting found so far), and:
  1. Compares harmonic-only vs harmonic+MFCC on random CV and show-independent F1.
  2. Reports permutation feature importance on the show-independent test set
     (Friends, entirely unseen during training) for the combined model.
"""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HARMONIC_PATH = "data/harmonic_features.csv"
MFCC_PATH = "data/mfcc_features.csv"
NON_FEATURE_COLS = {"id", "sarcasm", "show", "speaker"}


def load_combined() -> pd.DataFrame:
    harmonic = pd.read_csv(HARMONIC_PATH)
    mfcc = pd.read_csv(MFCC_PATH)
    return harmonic.merge(mfcc, on="id", how="inner")


def normalize_by_show(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    df = df.copy()
    for c in feature_cols:
        df[c] = df[c].fillna(df[c].mean())
        group_mean = df.groupby("show")[c].transform("mean")
        group_std = df.groupby("show")[c].transform("std").fillna(1).replace(0, 1)
        df[c] = (df[c] - group_mean) / group_std
    return df


def make_clf() -> SVC:
    return make_pipeline(StandardScaler(), SVC(C=10.0, gamma="scale", kernel="rbf"))


def evaluate(df: pd.DataFrame, feature_cols: list, label: str) -> None:
    df_norm = normalize_by_show(df, feature_cols)
    X = df_norm[feature_cols].values
    y = df_norm["sarcasm"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    f1s = []
    for train_idx, test_idx in skf.split(X, y):
        clf = make_clf().fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        report = classification_report(y[test_idx], pred, output_dict=True, digits=3)
        f1s.append(report["weighted avg"]["f1-score"])
    random_cv_f1 = np.mean(f1s)

    is_friends = (df_norm["show"] == "FRIENDS").values
    train_idx, test_idx = ~is_friends, is_friends
    clf = make_clf().fit(X[train_idx], y[train_idx])
    pred = clf.predict(X[test_idx])
    si_f1 = classification_report(y[test_idx], pred, output_dict=True, digits=3)["weighted avg"]["f1-score"]

    print(f"[{label}] n_features={len(feature_cols)}  random-CV F1={random_cv_f1:.3f}  show-independent F1={si_f1:.3f}")
    return clf, X, y, train_idx, test_idx


def main() -> None:
    df = load_combined()
    harmonic_cols = [c for c in pd.read_csv(HARMONIC_PATH).columns if c not in NON_FEATURE_COLS]
    mfcc_cols = [c for c in pd.read_csv(MFCC_PATH).columns if c != "id"]
    combined_cols = harmonic_cols + mfcc_cols

    evaluate(df, harmonic_cols, "harmonic only")
    clf, X, y, train_idx, test_idx = evaluate(df, combined_cols, "harmonic + MFCC")

    print()
    print("=== Permutation feature importance on show-independent test set (Friends) ===")
    result = permutation_importance(clf, X[test_idx], y[test_idx], n_repeats=30, random_state=0,
                                     scoring="f1_weighted")
    order = np.argsort(result.importances_mean)[::-1]
    for i in order[:15]:
        print(f"{combined_cols[i]:<24} {result.importances_mean[i]:.4f} (+/- {result.importances_std[i]:.4f})")


if __name__ == "__main__":
    main()
