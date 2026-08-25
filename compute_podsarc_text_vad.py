#!/usr/bin/env python
"""Text-side V/A/D for PodSarc's 700-item sample, via the same NRC-VAD
Lexicon used for MUStARD (compute_text_vad.py). Requires the raw PodSarc
annotation JSON (llms_annotations_with_human_check.json) for the text field.
"""
import json

import pandas as pd

from compute_text_vad import load_lexicon, score_text

PODSARC_JSON = "/Users/minjoolee/Downloads/MUStARD/PodSarc/llms_annotations_with_human_check.json"
SAMPLE_PATH = "data/podsarc_labels_sample.csv"
OUTPUT_PATH = "data/podsarc_vad_scores.csv"


def main() -> None:
    lexicon = load_lexicon()
    with open(PODSARC_JSON) as file:
        data = json.load(file)
    text_by_nid = {item["nid"]: item["text"] for item in data}

    sample = pd.read_csv(SAMPLE_PATH)
    rows = []
    for nid in sample.nid:
        v, a, d = score_text(text_by_nid.get(nid, ""), lexicon)
        rows.append({"nid": nid, "valence": v, "arousal": a, "dominance": d})

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)
    coverage = df[["valence", "arousal", "dominance"]].notna().all(axis=1).sum()
    print(f"Wrote {len(df)} rows to {OUTPUT_PATH} ({coverage} with lexicon coverage)")


if __name__ == "__main__":
    main()
