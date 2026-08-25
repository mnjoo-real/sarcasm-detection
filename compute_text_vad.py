#!/usr/bin/env python
"""Scores every MUStARD utterance's text on valence/arousal/dominance using
the NRC-VAD Lexicon (Mohammad 2018, data/vad-nrc-lexicon.csv), by averaging
word-level norms over the utterance. Output feeds analyze_audio_vad.py's
cross-modal gap analysis and the standalone text-VAD correlation check.
"""
import json
import re

import pandas as pd

LEXICON_PATH = "data/vad-nrc-lexicon.csv"
DATA_PATH = "data/sarcasm_data.json"
OUTPUT_PATH = "data/vad_scores.csv"


def load_lexicon() -> dict:
    lex = pd.read_csv(LEXICON_PATH, keep_default_na=False, na_values=[""])
    return {row.Word.lower(): (row.valence, row.arousal, row.dominance) for row in lex.itertuples()}


def score_text(text: str, lexicon: dict):
    words = re.findall(r"[a-zA-Z']+", text.lower())
    vals = [lexicon[w] for w in words if w in lexicon]
    if not vals:
        return None, None, None
    v = sum(x[0] for x in vals) / len(vals)
    a = sum(x[1] for x in vals) / len(vals)
    d = sum(x[2] for x in vals) / len(vals)
    return v, a, d


def main() -> None:
    lexicon = load_lexicon()
    with open(DATA_PATH) as file:
        data = json.load(file)

    rows = []
    for id_, entry in data.items():
        v, a, d = score_text(entry["utterance"], lexicon)
        rows.append({"id": id_, "valence": v, "arousal": a, "dominance": d})

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)
    coverage = df[["valence", "arousal", "dominance"]].notna().all(axis=1).sum()
    print(f"Wrote {len(df)} rows to {OUTPUT_PATH} ({coverage} with lexicon coverage)")


if __name__ == "__main__":
    main()
