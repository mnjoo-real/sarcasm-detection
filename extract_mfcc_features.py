#!/usr/bin/env python
"""Extracts MFCC (+ spectral centroid) features per utterance, to be combined
with the harmonic/prosodic features in data/harmonic_features.csv.

Uses librosa (only available in the py3.12 venv, since librosa's numba
dependency has no wheel for py3.14 on this machine).
"""
import argparse
import os

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

N_MFCC = 13


def extract_mfcc(path: str) -> dict:
    y, sr = librosa.load(path, sr=None, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    features = {}
    for i in range(N_MFCC):
        features[f"mfcc{i+1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc{i+1}_std"] = float(np.std(mfcc[i]))
    features["spectral_centroid_mean"] = float(np.mean(centroid))
    features["spectral_centroid_std"] = float(np.std(centroid))
    return features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--output", default="data/mfcc_features.csv")
    args = parser.parse_args()

    rows = []
    for filename in tqdm(sorted(os.listdir(args.audio_dir)), desc="Extracting MFCC features"):
        if not filename.endswith(".wav"):
            continue
        id_ = filename[:-4]
        try:
            features = extract_mfcc(os.path.join(args.audio_dir, filename))
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
