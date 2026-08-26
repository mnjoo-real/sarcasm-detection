#!/usr/bin/env python
"""Extracts log-mel-spectrogram summary features (mean+std per mel band, like
MFCC's mean+std per coefficient) per utterance, to be combined with the other
feature tables. 40 mel bands (a standard choice for speech feature summaries,
as opposed to the ~128 bands typically used for image-style CNN input) keep
this comparable in size to MFCC's 28 dims rather than an unwieldy few hundred.
"""
import argparse
import os

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

N_MELS = 40
FMAX = 8000


def extract_melspec(path: str) -> dict:
    y, sr = librosa.load(path, sr=None, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=min(FMAX, sr // 2))
    log_S = librosa.power_to_db(S)

    features = {}
    for i in range(N_MELS):
        features[f"mel{i+1}_mean"] = float(np.mean(log_S[i]))
        features[f"mel{i+1}_std"] = float(np.std(log_S[i]))
    return features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--output", default="data/melspec_features.csv")
    args = parser.parse_args()

    rows = []
    for filename in tqdm(sorted(os.listdir(args.audio_dir)), desc="Extracting mel-spectrogram features"):
        if not filename.endswith(".wav"):
            continue
        id_ = filename[:-4]
        try:
            features = extract_melspec(os.path.join(args.audio_dir, filename))
        except Exception as e:
            print(f"Failed on {id_}: {e}")
            continue
        features["id"] = id_
        rows.append(features)

    df = pd.DataFrame(rows).set_index("id")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output)
    print(f"Wrote {len(df)} rows x {len(df.columns)} columns to {args.output}")


if __name__ == "__main__":
    main()
