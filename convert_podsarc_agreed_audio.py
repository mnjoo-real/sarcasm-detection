#!/usr/bin/env python
"""Converts the 8139 'LLM-agreed' PodSarc utterances' mp3 audio (from
all_processed/) to wav in data/podsarc_audios/, matching the existing
disagreed-set wavs' format (22050 Hz mono PCM16), so extract_podsarc_features.py
can be pointed at the same audio-dir for both subsets. Skips files that
already exist, so it's safe to interrupt and resume.
"""
import subprocess

import imageio_ffmpeg
import pandas as pd
from tqdm import tqdm

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
LABELS = "data/podsarc_labels_agreed.csv"
AUDIO_DIR = "data/podsarc_audios"


def main() -> None:
    df = pd.read_csv(LABELS)
    ok, failed = 0, []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="converting mp3->wav"):
        out_path = f"{AUDIO_DIR}/{row['nid']}.wav"
        import os
        if os.path.exists(out_path):
            ok += 1
            continue
        result = subprocess.run(
            [FFMPEG, "-y", "-i", row["audio_path"], "-ar", "22050", "-ac", "1", out_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            failed.append((row["nid"], result.stderr[-300:]))
        else:
            ok += 1

    print(f"Done. {ok} ok, {len(failed)} failed.")
    if failed:
        print(failed[:10])


if __name__ == "__main__":
    main()
