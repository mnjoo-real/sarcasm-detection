#!/usr/bin/env python
"""3-axis (direct/ironic x friendly/hostile x proportionate/hyperbolic)
analysis of PodSarc's 287 sarcastic utterances, using podsarc_quadrant_labels.py.

The third axis was added because PodSarc's "sarcasm" turned out to be
dominated by direct, non-hostile utterances (77% under the original 2-axis
scheme) -- much of it hyperbolic exaggeration rather than classic
literal-meaning-flip irony, a distinction the 2-axis scheme couldn't capture.
"""
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

from podsarc_quadrant_labels import LABELS


def octant_of(direct_ironic: int, friendly_hostile: int, proportionate_hyperbolic: int) -> str:
    d = "direct" if direct_ironic == 1 else "ironic"
    f = "friendly" if friendly_hostile >= 0 else "hostile"
    p = "proportionate" if proportionate_hyperbolic == 1 else "hyperbolic"
    return f"{d}+{f}+{p}"


def main() -> None:
    rows = [{"nid": nid, "direct_ironic": di, "friendly_hostile": fh, "proportionate_hyperbolic": ph,
             "octant": octant_of(di, fh, ph)} for nid, (di, fh, ph) in LABELS.items()]
    labels_df = pd.DataFrame(rows)
    print(labels_df["octant"].value_counts())
    labels_df.to_csv("data/podsarc_octant_labels.csv", index=False)

    harmonic = pd.read_csv("data/podsarc_harmonic_features.csv")
    podlabels = pd.read_csv("data/podsarc_labels.csv")[["nid", "episode"]]
    df = labels_df.merge(harmonic, on="nid").merge(podlabels, on="nid")
    print("\nmerged n =", len(df))

    key_feats = [
        "f0_mean", "f0_std", "f0_range", "hnr_mean", "voiced_ratio", "speaking_rate_proxy",
        "energy_std", "dissonance_mean", "melodic_path_length", "direction_change_rate",
        "stress_contrast", "ioi_cv", "rpvi_pause_filtered", "wholetone_fit", "tonal_clarity",
        "pitch_stress_contrast",
    ]

    g_prop = df[df.proportionate_hyperbolic == 1]
    g_hyp = df[df.proportionate_hyperbolic == -1]
    print(f"proportionate n={len(g_prop)}, hyperbolic n={len(g_hyp)}")
    print(f'\n{"feature":<20}{"median_prop":>12}{"median_hyp":>12}{"pooled_p":>10}{"ANCOVA_p":>10}')
    for c in key_feats:
        u, p_pooled = stats.mannwhitneyu(g_prop[c].dropna(), g_hyp[c].dropna())
        model = smf.ols(f"{c} ~ proportionate_hyperbolic + C(episode)", data=df).fit()
        p_anc = model.pvalues.get("proportionate_hyperbolic", float("nan"))
        print(f"{c:<20}{g_prop[c].median():>12.3f}{g_hyp[c].median():>12.3f}{p_pooled:>10.4f}{p_anc:>10.4f}")


if __name__ == "__main__":
    main()
