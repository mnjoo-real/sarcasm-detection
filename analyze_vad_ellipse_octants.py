#!/usr/bin/env python
"""Ellipse-region method applied to PodSarc's 3-axis sarcasm-subtype octants
(podsarc_quadrant_labels.py) in wav2vec2 V/A/D space. Only octants with
enough points to fit a covariance ellipse are plotted (the hostile ones are
too sparse: 8, 2, 1, and 0 items -- see analyze_podsarc_octants.py).
"""
import itertools

import numpy as np
import pandas as pd

from analyze_vad_ellipse import plot_ellipses
from podsarc_quadrant_labels import LABELS

MIN_N = 15


def octant_of(direct_ironic: int, friendly_hostile: int, proportionate_hyperbolic: int) -> str:
    d = "direct" if direct_ironic == 1 else "ironic"
    f = "friendly" if friendly_hostile >= 0 else "hostile"
    p = "proportionate" if proportionate_hyperbolic == 1 else "hyperbolic"
    return f"{d}+{f}+{p}"


def main() -> None:
    rows = [{"nid": nid, "octant": octant_of(di, fh, ph)} for nid, (di, fh, ph) in LABELS.items()]
    labels_df = pd.DataFrame(rows)

    vad = pd.read_csv("data/podsarc_audio_vad.csv")
    df = labels_df.merge(vad, on="nid", how="inner")
    print(f"n={len(df)}")
    counts = df["octant"].value_counts()
    print(counts)

    keep = counts[counts >= MIN_N].index
    df_plot = df[df["octant"].isin(keep)]
    print(f"\nplotting {len(keep)} octants with n>={MIN_N}: {list(keep)}")

    plot_ellipses(df_plot, "audio_valence_wav2vec", "audio_arousal_wav2vec", "octant",
                  "PodSarc sarcasm subtypes in Valence-Arousal space",
                  "data/podsarc_octant_vad_ellipse.png", k=1.0)

    print("\n=== Centroid distances between octants (Euclidean, V-A-D space) ===")
    vad_cols = ["audio_valence_wav2vec", "audio_arousal_wav2vec", "audio_dominance_wav2vec"]
    centroids = df_plot.groupby("octant")[vad_cols].mean()
    print(centroids)
    for a, b in itertools.combinations(centroids.index, 2):
        dist = np.linalg.norm(centroids.loc[a] - centroids.loc[b])
        print(f"  {a} <-> {b}: {dist:.4f}")


if __name__ == "__main__":
    main()
