#!/usr/bin/env python
"""Tests whether adding the episode-control-validated formant/voice-quality
features (from analyze_formant_voice_quality.py --corpus podsarc) improves
on PodSarc's own validated-15 feature set, via the same GroupKFold(episode)
protocol as train_podsarc_classifier.py.
"""
import pandas as pd

from train_podsarc_classifier import (
    PODSARC_VALIDATED_15, normalize_by_episode, evaluate,
)

# Survived episode-ANCOVA in analyze_formant_voice_quality.py --corpus podsarc
VALIDATED_FORMANT_VQ = ["f1_mean", "f3_mean", "cpp_mean", "cpp_std", "h1h2_std"]


def main() -> None:
    harmonic = pd.read_csv("data/podsarc_harmonic_features.csv")
    labels = pd.read_csv("data/podsarc_labels.csv")[["nid", "episode", "sarcasm"]]
    df = harmonic.merge(labels, on="nid", how="inner")
    all_cols = [c for c in harmonic.columns if c != "nid"]
    normed = normalize_by_episode(df, all_cols)

    evaluate(normed, PODSARC_VALIDATED_15, "PodSarc-validated 15")
    evaluate(normed, PODSARC_VALIDATED_15 + VALIDATED_FORMANT_VQ, "15 + 5 validated formant/VQ = 20")
    for feat in VALIDATED_FORMANT_VQ:
        evaluate(normed, PODSARC_VALIDATED_15 + [feat], f"15 + only {feat}")


if __name__ == "__main__":
    main()
