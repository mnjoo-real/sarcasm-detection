#!/usr/bin/env python
"""Models sarcastic/non-sarcastic classes as ellipse regions in wav2vec2
audio V/A/D space, following Han & Cha (2017)'s extension of Russell's
Circumplex Model: represent each class as a bivariate-Gaussian confidence
ellipse (mean + covariance -> rotation angle from correlation, semi-axes
from per-axis std) instead of a single point, then classify via the
Bayesian/quadratic decision boundary between ellipses.

This is mathematically Quadratic Discriminant Analysis (QDA): a class-
conditional Gaussian per class, Bayes' rule for the decision boundary. We
fit it with sklearn's QDA (and LDA, which assumes one shared covariance,
as a simpler baseline) and evaluate with the same show/episode-grouped
held-out protocol used throughout this project, not simple in-sample
accuracy, since the paper's own 92.86% figure is in-sample.
"""
import argparse

import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupKFold


def ellipse_params(points: np.ndarray, k: float = 1.0):
    """Returns (center, width, height, angle_deg) for the k-sigma confidence
    ellipse of a 2D point cloud, matching the paper's Eq. 2-4 exactly:
    rotation from the correlation coefficient, semi-axes from k*std along
    the principal axes (eigenvectors of the covariance matrix)."""
    mean = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * k * np.sqrt(eigvals)
    return mean, width, height, angle


def plot_ellipses(df: pd.DataFrame, x_col: str, y_col: str, group_col: str,
                   title: str, out_path: str, k: float = 1.5) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.tab10.colors
    for i, (label, sub) in enumerate(df.groupby(group_col)):
        points = sub[[x_col, y_col]].dropna().values
        if len(points) < 3:
            continue
        color = colors[i % len(colors)]
        ax.scatter(points[:, 0], points[:, 1], s=8, alpha=0.35, color=color, label=f"{label} (n={len(points)})")
        mean, width, height, angle = ellipse_params(points, k=k)
        ell = Ellipse(mean, width, height, angle=angle, facecolor="none",
                      edgecolor=color, linewidth=2.5)
        ax.add_patch(ell)
        ax.plot(*mean, marker="x", color=color, markersize=10, markeredgewidth=2)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"saved plot to {out_path}")


def evaluate_qda_lda(df: pd.DataFrame, feature_cols: list, group_col: str) -> None:
    X = df[feature_cols].values
    y = df["sarcasm"].values
    groups = df[group_col].values

    n_splits = min(5, len(np.unique(groups)))
    for name, clf_cls in [("LDA", LinearDiscriminantAnalysis), ("QDA", QuadraticDiscriminantAnalysis)]:
        gkf = GroupKFold(n_splits=n_splits)
        f1s = []
        for tr, te in gkf.split(X, y, groups):
            clf = clf_cls()
            clf.fit(X[tr], y[tr])
            pred = clf.predict(X[te])
            rep = classification_report(y[te], pred, output_dict=True, zero_division=0)
            f1s.append(rep["weighted avg"]["f1-score"])
        print(f"{name} ({group_col}-grouped 5-fold): weighted F1 = {np.mean(f1s):.3f} (+/- {np.std(f1s):.3f})")

    # In-sample accuracy too, for a direct comparison to the paper's own
    # reported figure (which is also in-sample, not held out).
    clf = QuadraticDiscriminantAnalysis().fit(X, y)
    acc = (clf.predict(X) == y).mean()
    print(f"QDA in-sample accuracy (paper-style, optimistic): {acc:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=["mustard", "podsarc"], default="mustard")
    args = parser.parse_args()

    vad_cols = ["audio_arousal_wav2vec", "audio_valence_wav2vec", "audio_dominance_wav2vec"]

    if args.corpus == "mustard":
        vad = pd.read_csv("data/audio_vad.csv")
        labels = pd.read_csv("data/harmonic_features.csv")[["id", "sarcasm", "show"]]
        df = vad.merge(labels, on="id", how="inner")
        group_col = "show"
    else:
        vad = pd.read_csv("data/podsarc_audio_vad.csv")
        labels = pd.read_csv("data/podsarc_labels_sample.csv")[["nid", "sarcasm", "episode"]]
        df = vad.merge(labels, on="nid", how="inner")
        group_col = "episode"

    df["sarcasm_label"] = df["sarcasm"].map({0: "non-sarcastic", 1: "sarcastic"})
    print(f"n={len(df)}")

    plot_ellipses(df, "audio_valence_wav2vec", "audio_arousal_wav2vec", "sarcasm_label",
                  f"{args.corpus}: sarcastic vs non-sarcastic in Valence-Arousal space",
                  f"data/{args.corpus}_vad_ellipse.png")

    print("\n=== held-out classification via ellipse-equivalent models (LDA/QDA) ===")
    evaluate_qda_lda(df, vad_cols, group_col)


if __name__ == "__main__":
    main()
