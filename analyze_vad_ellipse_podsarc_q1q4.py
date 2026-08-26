#!/usr/bin/env python
"""Ellipse-region method applied to PodSarc collapsed to the original 2-axis
Q1-Q4 scheme (direct/ironic x friendly/hostile only, ignoring the 3rd
proportionate/hyperbolic axis) -- for direct comparison against MUStARD's
Q1-Q4 plot using the same axes.
"""
import itertools

import numpy as np
import pandas as pd

from analyze_vad_ellipse import plot_ellipses
from podsarc_quadrant_labels import LABELS

MIN_N = 15


def quadrant_of(direct_ironic: int, friendly_hostile: int) -> str:
    if direct_ironic == 1 and friendly_hostile >= 0:
        return "Q1_praise"
    if direct_ironic == 1 and friendly_hostile < 0:
        return "Q2_criticism"
    if direct_ironic == -1 and friendly_hostile < 0:
        return "Q3_bite"
    return "Q4_humor"


def main() -> None:
    rows = [{"nid": nid, "quadrant": quadrant_of(di, fh)} for nid, (di, fh, ph) in LABELS.items()]
    labels_df = pd.DataFrame(rows)

    vad = pd.read_csv("data/podsarc_audio_vad.csv")
    df = labels_df.merge(vad, on="nid", how="inner")
    print(f"n={len(df)}")
    counts = df["quadrant"].value_counts()
    print(counts)

    keep = counts[counts >= MIN_N].index
    df_plot = df[df["quadrant"].isin(keep)]
    print(f"\nplotting quadrants with n>={MIN_N}: {list(keep)}")

    plot_ellipses(df_plot, "audio_valence_wav2vec", "audio_arousal_wav2vec", "quadrant",
                  "PodSarc sarcasm subtypes (Q1-Q4) in Valence-Arousal space",
                  "data/podsarc_q1q4_vad_ellipse.png", k=1.0)

    print("\n=== Centroid distances between quadrants (Euclidean, V-A-D space) ===")
    vad_cols = ["audio_valence_wav2vec", "audio_arousal_wav2vec", "audio_dominance_wav2vec"]
    centroids = df_plot.groupby("quadrant")[vad_cols].mean()
    print(centroids)
    for a, b in itertools.combinations(centroids.index, 2):
        dist = np.linalg.norm(centroids.loc[a] - centroids.loc[b])
        print(f"  {a} <-> {b}: {dist:.4f}")


if __name__ == "__main__":
    main()
