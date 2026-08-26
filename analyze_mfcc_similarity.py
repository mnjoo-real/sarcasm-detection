#!/usr/bin/env python
"""Cosine-similarity comparison of sarcastic vs. non-sarcastic MFCC profiles,
for either corpus (MUStARD or PodSarc). Reports:

1. Centroid similarity: cosine similarity between the two classes' mean MFCC
   vectors (a single-number summary of how similar the "average voice" of
   each class looks in MFCC space).
2. Intra- vs inter-class pairwise similarity: mean cosine similarity within
   each class vs. across classes (sampled if the corpus is large). If a
   class is acoustically coherent and separable, intra-class similarity
   should exceed inter-class similarity.

Run on both raw MFCC vectors and group-normalized ones (show/episode
z-scored), since MFCC is known to carry speaker/recording-baseline
information that a raw cosine similarity would otherwise be dominated by.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

MFCC_COLS = [
    f"mfcc{i}_{stat}" for i in range(1, 14) for stat in ("mean", "std")
] + ["spectral_centroid_mean", "spectral_centroid_std"]


def normalize_by_group(df: pd.DataFrame, cols: list, group_col: str) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = df[c].fillna(df[c].mean())
        group_mean = df.groupby(group_col)[c].transform("mean")
        group_std = df.groupby(group_col)[c].transform("std").fillna(1).replace(0, 1)
        df[c] = (df[c] - group_mean) / group_std
    return df


def report(df: pd.DataFrame, cols: list, label: str, max_pairs_per_class: int = 2000,
           seed: int = 0) -> None:
    X = df[cols].values
    y = df["sarcasm"].values

    centroid0 = X[y == 0].mean(axis=0, keepdims=True)
    centroid1 = X[y == 1].mean(axis=0, keepdims=True)
    centroid_sim = cosine_similarity(centroid0, centroid1)[0, 0]

    rng = np.random.default_rng(seed)

    def sampled_mean_sim(A, B, max_pairs):
        if len(A) * len(B) <= max_pairs:
            sims = cosine_similarity(A, B)
            return sims.mean()
        idx_a = rng.integers(0, len(A), size=max_pairs)
        idx_b = rng.integers(0, len(B), size=max_pairs)
        sims = np.sum(A[idx_a] * B[idx_b], axis=1) / (
            np.linalg.norm(A[idx_a], axis=1) * np.linalg.norm(B[idx_b], axis=1))
        return sims.mean()

    X0, X1 = X[y == 0], X[y == 1]
    intra0 = sampled_mean_sim(X0, X0, max_pairs_per_class)
    intra1 = sampled_mean_sim(X1, X1, max_pairs_per_class)
    inter = sampled_mean_sim(X0, X1, max_pairs_per_class)

    print(f"--- {label} (n0={len(X0)}, n1={len(X1)}) ---")
    print(f"  centroid cosine similarity (sarc vs non-sarc):    {centroid_sim:.4f}")
    print(f"  mean intra-class similarity, non-sarcastic:       {intra0:.4f}")
    print(f"  mean intra-class similarity, sarcastic:           {intra1:.4f}")
    print(f"  mean inter-class similarity:                      {inter:.4f}")
    print(f"  separability gap (avg(intra) - inter):            {(intra0+intra1)/2 - inter:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=["mustard", "podsarc"], required=True)
    args = parser.parse_args()

    if args.corpus == "mustard":
        mfcc = pd.read_csv("data/mfcc_features.csv")
        labels = pd.read_csv("data/harmonic_features.csv")[["id", "sarcasm", "show"]]
        df = mfcc.merge(labels, on="id", how="inner")
        group_col = "show"
    else:
        # extract_mfcc_features.py always names the id column "id" (even for
        # PodSarc, whose own id column is "nid" elsewhere in this project).
        mfcc = pd.read_csv("data/podsarc_mfcc_features.csv").rename(columns={"id": "nid"})
        labels = pd.read_csv("data/podsarc_labels.csv")[["nid", "sarcasm", "episode"]]
        df = mfcc.merge(labels, on="nid", how="inner")
        group_col = "episode"

    report(df, MFCC_COLS, f"{args.corpus} raw MFCC")
    normed = normalize_by_group(df, MFCC_COLS, group_col)
    report(normed, MFCC_COLS, f"{args.corpus} {group_col}-normalized MFCC")


if __name__ == "__main__":
    main()
