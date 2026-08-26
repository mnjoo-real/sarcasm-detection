#!/usr/bin/env python
"""Statistical validation of the newest acoustic features -- formants
(F1/F2/F3 + dispersion) and voice quality (CPP, H1-H2 spectral tilt) --
against sarcastic vs. non-sarcastic labels, in both corpora. Same protocol
used for every prior feature batch: pooled Mann-Whitney first, then an
ANCOVA controlling for show/episode to rule out confounds before trusting
a pooled-significant result.
"""
import argparse

import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

NEW_FEATURES = [
    "f1_mean", "f1_std", "f2_mean", "f2_std", "f3_mean", "f3_std", "formant_dispersion",
    "cpp_mean", "cpp_std", "h1h2_mean", "h1h2_std",
]


def run(df: pd.DataFrame, group_col: str) -> None:
    g0, g1 = df[df.sarcasm == 0], df[df.sarcasm == 1]
    print(f"n={len(df)} (non-sarcastic={len(g0)}, sarcastic={len(g1)}), "
          f"{group_col}s={df[group_col].nunique()}")

    print(f"\n{'feature':<20}{'median_non':>12}{'median_sarc':>12}{'pooled_p':>10}{'ANCOVA_p':>10}")
    pooled_sig, survives = [], []
    df[group_col] = df[group_col].astype(str)
    for c in NEW_FEATURES:
        _, p_pooled = stats.mannwhitneyu(g0[c].dropna(), g1[c].dropna())
        model = smf.ols(f"{c} ~ sarcasm + C({group_col})", data=df).fit()
        p_ancova = model.pvalues.get("sarcasm", float("nan"))
        flag = ""
        if p_pooled < 0.05:
            pooled_sig.append(c)
            flag = "*" if p_ancova < 0.05 else "(vanishes)"
            if p_ancova < 0.05:
                survives.append(c)
        print(f"{c:<20}{g0[c].median():>12.3f}{g1[c].median():>12.3f}{p_pooled:>10.4f}{p_ancova:>10.4f}  {flag}")

    print(f"\npooled-significant: {len(pooled_sig)}/{len(NEW_FEATURES)} -> {pooled_sig}")
    print(f"survives {group_col}-control: {len(survives)}/{len(pooled_sig) if pooled_sig else 0} -> {survives}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=["mustard", "podsarc"], default="mustard")
    args = parser.parse_args()

    if args.corpus == "mustard":
        df = pd.read_csv("data/harmonic_features.csv")
        run(df, "show")
    else:
        harmonic = pd.read_csv("data/podsarc_harmonic_features.csv")
        labels = pd.read_csv("data/podsarc_labels.csv")[["nid", "episode", "sarcasm"]]
        df = harmonic.merge(labels, on="nid", how="inner")
        run(df, "episode")


if __name__ == "__main__":
    main()
