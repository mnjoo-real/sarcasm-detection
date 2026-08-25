#!/usr/bin/env python
"""Two checks on whether MUStARD's acoustic sarcasm markers hold up in a
different domain (podcast conversation vs. scripted TV comedy):

1. Direction-replication: for each of MUStARD's 20 final features, does
   sarcastic-vs-not point the same way in PodSarc (pooled Mann-Whitney)?
2. Full 33-feature re-screen: pooled + episode-controlled ANCOVA across
   every feature (including the 13 MUStARD rejected), since a feature MUStARD
   discarded is not guaranteed to be useless in a new domain.
"""
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

harmonic = pd.read_csv("data/podsarc_harmonic_features.csv")
labels = pd.read_csv("data/podsarc_labels.csv")[["nid", "episode", "sarcasm"]]
df = harmonic.merge(labels, on="nid", how="inner")
print(f"n={len(df)}, episodes={df['episode'].nunique()}")
print(df["sarcasm"].value_counts())

ALL_FEATURES = [c for c in harmonic.columns if c != "nid"]

# Direction MUStARD found for its final 20 features (sarcastic higher=+, lower=-)
MUSTARD_DIRECTION = {
    "dissonance_mean": "+", "dissonance_std": "+", "hnr_mean": "-", "f0_mean": "-",
    "speaking_rate_proxy": "+", "melodic_path_length": "-", "direction_change_rate": "-",
    "final_initial_diff": "+", "num_notes_per_sec": "-", "stress_contrast": "-",
}

g0, g1 = df[df.sarcasm == 0], df[df.sarcasm == 1]

print("\n=== Full 33-feature pooled screen ===")
print(f'{"feature":<24}{"median0":>10}{"median1":>10}{"MUStARD dir":>12}{"PodSarc p":>12}{"match?":>10}')
pooled_sig = []
for c in ALL_FEATURES:
    u, p = stats.mannwhitneyu(g0[c].dropna(), g1[c].dropna())
    podsarc_dir = "+" if g1[c].median() > g0[c].median() else "-"
    expected = MUSTARD_DIRECTION.get(c, "?")
    if p >= 0.05:
        match = "no-sig"
    elif expected == "?":
        match = "NEW-SIG"
    elif expected == podsarc_dir:
        match = "YES"
    else:
        match = "FLIP"
    if p < 0.05:
        pooled_sig.append(c)
    print(f"{c:<24}{g0[c].median():>10.3f}{g1[c].median():>10.3f}{expected:>12}{p:>12.4f}{match:>10}")

print(f"\n{len(pooled_sig)}/{len(ALL_FEATURES)} significant pooled in PodSarc")

print("\n=== Episode-controlled ANCOVA on pooled-significant features ===")
df["episode"] = df["episode"].astype(str)
survives = []
for c in pooled_sig:
    model = smf.ols(f"{c} ~ sarcasm + C(episode)", data=df).fit()
    p = model.pvalues["sarcasm"]
    print(f"{c:<24} ANCOVA_p={p:.4f}")
    if p < 0.05:
        survives.append(c)

print(f"\nsurvives episode control: {len(survives)}/{len(pooled_sig)}")
print(survives)
