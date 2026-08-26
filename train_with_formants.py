#!/usr/bin/env python
"""Tests whether adding the show-control-validated formant/voice-quality
features (from analyze_formant_voice_quality.py) improves on MUStARD's
final 23-feature model (train_final_model.py), via the same LOSO protocol.
"""
import pandas as pd

from train_final_model import FEATURE_COLS, HARMONIC_PATH, AUDIO_VAD_PATH, loso_eval

# Survived show-ANCOVA in analyze_formant_voice_quality.py --corpus mustard
VALIDATED_FORMANT_VQ = [
    "f1_mean", "f2_mean", "f3_mean", "f3_std", "formant_dispersion", "h1h2_mean", "h1h2_std",
]


def main() -> None:
    features = pd.read_csv(HARMONIC_PATH)
    audio_vad = pd.read_csv(AUDIO_VAD_PATH)
    df = features.merge(audio_vad, on="id", how="inner")

    print("=== baseline 23 features ===")
    loso_eval(df, FEATURE_COLS)

    print("\n=== 23 + 7 validated formant/voice-quality = 30 features ===")
    loso_eval(df, FEATURE_COLS + VALIDATED_FORMANT_VQ)

    for feat in VALIDATED_FORMANT_VQ:
        print(f"\n=== 23 + only {feat} ===")
        loso_eval(df, FEATURE_COLS + [feat])


if __name__ == "__main__":
    main()
