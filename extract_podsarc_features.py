#!/usr/bin/env python
"""Batch-extracts harmony/melody/rhythm features for PodSarc utterances, with
checkpointing: writes to the output CSV every --checkpoint-every items (append
mode) and skips nids already present in an existing output file on restart,
so an interrupted run loses at most one checkpoint interval of work.
"""
import argparse
import os

import pandas as pd
from tqdm import tqdm

from harmonic_features import extract_all_features

DEFAULT_LABELS = "data/podsarc_labels.csv"
DEFAULT_AUDIO_DIR = "data/podsarc_audios"
DEFAULT_OUTPUT = "data/podsarc_harmonic_features.csv"


def already_done(output_path: str) -> set:
    if not os.path.exists(output_path):
        return set()
    return set(pd.read_csv(output_path)["nid"].astype(str))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    args = parser.parse_args()

    df = pd.read_csv(args.labels)
    done = already_done(args.output)
    todo = [nid for nid in df.nid if str(nid) not in done]
    print(f"{len(done)} already done, {len(todo)} remaining")

    buffer = []
    failed = []
    header_needed = not os.path.exists(args.output)

    def flush():
        nonlocal buffer, header_needed
        if not buffer:
            return
        pd.DataFrame(buffer).to_csv(args.output, mode="a", index=False, header=header_needed)
        header_needed = False
        buffer = []

    for i, nid in enumerate(tqdm(todo, desc="PodSarc harmonic features")):
        path = os.path.join(args.audio_dir, f"{nid}.wav")
        try:
            feats = extract_all_features(path)
        except Exception as e:
            failed.append((nid, str(e)))
            continue
        feats["nid"] = nid
        buffer.append(feats)

        if (i + 1) % args.checkpoint_every == 0:
            flush()

    flush()
    print(f"Done. {len(todo) - len(failed)} succeeded this run, {len(failed)} failed.")
    if failed:
        print(failed[:10])


if __name__ == "__main__":
    main()
