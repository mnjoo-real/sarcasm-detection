#!/usr/bin/env python
"""Same ellipse-region method as analyze_vad_ellipse.py, but for MUStARD's
4 sarcasm-subtype quadrants (praise/criticism/bite/humor from
quadrant_labels.py) instead of the binary sarcastic/non-sarcastic split --
directly testing whether "bite" and "humor" occupy separable regions of
V/A/D space, the finer-grained question the ellipse method was proposed for.
"""
import pandas as pd

from analyze_vad_ellipse import plot_ellipses, ellipse_params
from quadrant_labels import LABELS


def quadrant_of(direct_ironic: int, friendly_hostile: int) -> str:
    if direct_ironic == 1 and friendly_hostile >= 0:
        return "Q1_praise"
    if direct_ironic == 1 and friendly_hostile < 0:
        return "Q2_criticism"
    if direct_ironic == -1 and friendly_hostile < 0:
        return "Q3_bite"
    return "Q4_humor"


def main() -> None:
    rows = [{"id": id_, "quadrant": quadrant_of(di, fh)} for id_, (di, fh) in LABELS.items()]
    labels_df = pd.DataFrame(rows)

    vad = pd.read_csv("data/audio_vad.csv")
    df = labels_df.merge(vad, on="id", how="inner")
    print(f"n={len(df)}")
    print(df["quadrant"].value_counts())

    plot_ellipses(df, "audio_valence_wav2vec", "audio_arousal_wav2vec", "quadrant",
                  "MUStARD sarcasm subtypes in Valence-Arousal space",
                  "data/mustard_quadrant_vad_ellipse.png", k=1.0)

    print("\n=== Centroid distances between quadrants (Euclidean, V-A-D space) ===")
    vad_cols = ["audio_valence_wav2vec", "audio_arousal_wav2vec", "audio_dominance_wav2vec"]
    centroids = df.groupby("quadrant")[vad_cols].mean()
    print(centroids)
    import itertools
    import numpy as np
    for a, b in itertools.combinations(centroids.index, 2):
        dist = np.linalg.norm(centroids.loc[a] - centroids.loc[b])
        print(f"  {a} <-> {b}: {dist:.4f}")


if __name__ == "__main__":
    main()
