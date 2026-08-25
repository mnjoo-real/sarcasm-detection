#!/usr/bin/env python
"""Tests whether sarcasm splits into two acoustically distinct subtypes:
Q3 "bite" (ironic + hostile) vs Q4 "humor" (ironic + friendly), using manual
direct/ironic x friendly/hostile judgments (quadrant_labels.py) made by
reading each of the 345 sarcastic utterances with its context.

Q1 (direct+friendly=praise) and Q2 (direct+hostile=criticism) are computed
too, mainly as a face-validity check that MUStARD's sarcasm=True set is
dominated by Q3/Q4 rather than Q1/Q2.
"""
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

from quadrant_labels import LABELS

KEY_FEATURES = [
    "dissonance_mean", "dissonance_std", "hnr_mean", "f0_mean", "speaking_rate_proxy",
    "melodic_path_length", "direction_change_rate", "final_initial_diff", "num_notes_per_sec",
    "stress_contrast", "ioi_cv", "rpvi_pause_filtered",
]


def quadrant_of(direct_ironic: int, friendly_hostile: int) -> str:
    if direct_ironic == 1 and friendly_hostile >= 0:
        return "Q1_praise"
    if direct_ironic == 1 and friendly_hostile < 0:
        return "Q2_criticism"
    if direct_ironic == -1 and friendly_hostile < 0:
        return "Q3_bite"
    return "Q4_humor"


def main() -> None:
    rows = [{"id": id_, "direct_ironic": di, "friendly_hostile": fh, "quadrant": quadrant_of(di, fh)}
            for id_, (di, fh) in LABELS.items()]
    labels_df = pd.DataFrame(rows)
    print(labels_df["quadrant"].value_counts())

    features = pd.read_csv("data/harmonic_features.csv")
    df = labels_df.merge(features, on="id", how="inner")
    df.to_csv("data/quadrant_merged.csv", index=False)
    print("\nquadrant x show:")
    print(df.groupby("quadrant")["show"].value_counts())

    q3, q4 = df[df.quadrant == "Q3_bite"], df[df.quadrant == "Q4_humor"]
    print(f"\n=== Q3 (bite, n={len(q3)}) vs Q4 (humor, n={len(q4)}): pooled vs show-controlled ===")
    print(f'{"feature":<24}{"median_Q3":>10}{"median_Q4":>10}{"pooled_p":>10}{"ANCOVA_p":>10}')
    sub = df[df.quadrant.isin(["Q3_bite", "Q4_humor"])].copy()
    sub["is_bite"] = (sub.quadrant == "Q3_bite").astype(int)
    for c in KEY_FEATURES:
        _, p_pooled = stats.mannwhitneyu(q3[c].dropna(), q4[c].dropna())
        model = smf.ols(f"{c} ~ is_bite + C(show)", data=sub).fit()
        p_ancova = model.pvalues.get("is_bite", float("nan"))
        print(f"{c:<24}{q3[c].median():>10.3f}{q4[c].median():>10.3f}{p_pooled:>10.4f}{p_ancova:>10.4f}")


if __name__ == "__main__":
    main()
