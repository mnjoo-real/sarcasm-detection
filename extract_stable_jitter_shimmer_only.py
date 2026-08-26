#!/usr/bin/env python
"""Lightweight incremental extraction of just the glide-excluded jitter/shimmer
features (jitter_shimmer_stable_features) for an existing PodSarc audio set.
Checkpointed the same way as extract_podsarc_features.py.
"""
import argparse
import os

import pandas as pd
import parselmouth
from tqdm import tqdm

from harmonic_features import jitter_shimmer_stable_features


def already_done(output_path: str) -> set:
    if not os.path.exists(output_path):
        return set()
    return set(pd.read_csv(output_path)["nid"].astype(str))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--audio-dir", default="data/podsarc_audios")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    args = parser.parse_args()

    df = pd.read_csv(args.labels)
    done = already_done(args.output)
    todo = [nid for nid in df.nid if str(nid) not in done]
    print(f"{len(done)} already done, {len(todo)} remaining")

    buffer, failed = [], []
    header_needed = not os.path.exists(args.output)

    def flush():
        nonlocal buffer, header_needed
        if not buffer:
            return
        pd.DataFrame(buffer).to_csv(args.output, mode="a", index=False, header=header_needed)
        header_needed = False
        buffer = []

    for i, nid in enumerate(tqdm(todo, desc="stable jitter/shimmer")):
        path = os.path.join(args.audio_dir, f"{nid}.wav")
        try:
            sound = parselmouth.Sound(path)
            feats = jitter_shimmer_stable_features(sound)
        except Exception as e:
            failed.append((nid, str(e)))
            continue
        feats["nid"] = nid
        buffer.append(feats)
        if (i + 1) % args.checkpoint_every == 0:
            flush()

    flush()
    print(f"Done. {len(todo) - len(failed)} succeeded, {len(failed)} failed.")
    if failed:
        print(failed[:10])


if __name__ == "__main__":
    main()
