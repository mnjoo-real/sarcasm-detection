#!/usr/bin/env python
"""wav2vec2 audio V/A/D extraction for PodSarc, with the same checkpointing
(append + resume-by-skipping-done-nids) as extract_podsarc_features.py.
Must run in .venv_vad (see extract_audio_vad.py for why).
"""
import argparse
import os

import pandas as pd
from tqdm import tqdm

from extract_audio_vad import load_model, predict

DEFAULT_LABELS = "data/podsarc_labels.csv"
DEFAULT_AUDIO_DIR = "data/podsarc_audios"
DEFAULT_OUTPUT = "data/podsarc_audio_vad.csv"


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

    processor, model = load_model()

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

    for i, nid in enumerate(tqdm(todo, desc="PodSarc wav2vec2 VAD")):
        path = os.path.join(args.audio_dir, f"{nid}.wav")
        try:
            a, d, v = predict(path, processor, model)
        except Exception as e:
            failed.append((nid, str(e)))
            continue
        buffer.append({"nid": nid, "audio_arousal_wav2vec": a,
                        "audio_dominance_wav2vec": d, "audio_valence_wav2vec": v})

        if (i + 1) % args.checkpoint_every == 0:
            flush()

    flush()
    print(f"Done. {len(todo) - len(failed)} succeeded this run, {len(failed)} failed.")
    if failed:
        print(failed[:10])


if __name__ == "__main__":
    main()
