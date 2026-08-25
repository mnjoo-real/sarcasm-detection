#!/usr/bin/env python
"""Batch-extracts prosody/rhythm/harmony features for every MUStARD utterance
and merges them with the sarcasm label from data/sarcasm_data.json.

Usage:
    python extract_harmonic_features.py \
        --audio-dir data/audios/utterances_final \
        --output data/harmonic_features.csv
"""
import argparse
import json
import os

import pandas as pd
from tqdm import tqdm

from harmonic_features import extract_all_features

DATA_JSON = "data/sarcasm_data.json"
AUDIO_EXTENSIONS = (".wav", ".aac", ".mp3", ".flac")


def find_audio_file(audio_dir: str, id_: str) -> str:
    for ext in AUDIO_EXTENSIONS:
        path = os.path.join(audio_dir, id_ + ext)
        if os.path.exists(path):
            return path
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", required=True, help="Folder with one audio file per utterance id")
    parser.add_argument("--data-json", default=DATA_JSON)
    parser.add_argument("--output", default="data/harmonic_features.csv")
    args = parser.parse_args()

    with open(args.data_json) as file:
        dataset = json.load(file)

    rows = []
    missing = []
    for id_, entry in tqdm(dataset.items(), desc="Extracting harmonic features"):
        path = find_audio_file(args.audio_dir, id_)
        if not path:
            missing.append(id_)
            continue
        try:
            features = extract_all_features(path)
        except Exception as e:
            print(f"Failed on {id_}: {e}")
            continue
        features["id"] = id_
        features["sarcasm"] = int(entry["sarcasm"])
        features["show"] = entry["show"]
        rows.append(features)

    if missing:
        print(f"Skipped {len(missing)} utterances with no audio file found in {args.audio_dir}")

    df = pd.DataFrame(rows).set_index("id")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output)
    print(f"Wrote {len(df)} rows x {len(df.columns)} columns to {args.output}")


if __name__ == "__main__":
    main()
