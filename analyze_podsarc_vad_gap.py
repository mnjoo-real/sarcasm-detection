#!/usr/bin/env python
"""Cross-modal audio-vs-text V/A/D gap analysis for PodSarc (replicating the
MUStARD version in analyze_audio_vad.py, on the 700-item sample), plus the
sarcasm-vs-VAD group comparison itself.
"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

audio_vad = pd.read_csv("data/podsarc_audio_vad.csv")
text_vad = pd.read_csv("data/podsarc_vad_scores.csv")
labels = pd.read_csv("data/podsarc_labels_sample.csv")[["nid", "episode", "sarcasm"]]

df = audio_vad.merge(text_vad, on="nid", how="inner").merge(labels, on="nid", how="inner")
df = df.dropna(subset=["valence", "arousal", "dominance"])
print("n =", len(df))

audio_dims = ["audio_arousal_wav2vec", "audio_dominance_wav2vec", "audio_valence_wav2vec"]

print("\n=== Sarcastic vs non-sarcastic: real audio V/A/D ===")
g0, g1 = df[df.sarcasm == 0], df[df.sarcasm == 1]
for c in audio_dims:
    u, p_pooled = stats.mannwhitneyu(g0[c].dropna(), g1[c].dropna())
    model = smf.ols(f"{c} ~ sarcasm + C(episode)", data=df).fit()
    p_anc = model.pvalues["sarcasm"]
    print(f"{c:<26} median0={g0[c].median():.3f} median1={g1[c].median():.3f}  "
          f"pooled_p={p_pooled:.4f}  ANCOVA_p={p_anc:.4f}")


def z(s):
    return (s - s.mean()) / s.std()


df["audio_arousal_z"] = z(df.audio_arousal_wav2vec)
df["audio_valence_z"] = z(df.audio_valence_wav2vec)
df["audio_dominance_z"] = z(df.audio_dominance_wav2vec)
df["text_arousal_z"] = z(df.arousal)
df["text_valence_z"] = z(df.valence)
df["text_dominance_z"] = z(df.dominance)

df["gap_arousal"] = (df.audio_arousal_z - df.text_arousal_z).abs()
df["gap_valence"] = (df.audio_valence_z - df.text_valence_z).abs()
df["gap_dominance"] = (df.audio_dominance_z - df.text_dominance_z).abs()
df["gap_total"] = np.sqrt(df.gap_arousal**2 + df.gap_valence**2 + df.gap_dominance**2)
df.to_csv("data/podsarc_vad_gap.csv", index=False)

print("\n=== Cross-modal gap: sarcastic vs non-sarcastic ===")
g0, g1 = df[df.sarcasm == 0], df[df.sarcasm == 1]
print(f'{"gap":<14}{"median0":>10}{"median1":>10}{"pooled_p":>10}{"ANCOVA_p":>10}')
for c in ["gap_arousal", "gap_valence", "gap_dominance", "gap_total"]:
    u, p_pooled = stats.mannwhitneyu(g0[c].dropna(), g1[c].dropna())
    model = smf.ols(f"{c} ~ sarcasm + C(episode)", data=df).fit()
    p_anc = model.pvalues["sarcasm"]
    print(f"{c:<14}{g0[c].median():>10.3f}{g1[c].median():>10.3f}{p_pooled:>10.4f}{p_anc:>10.4f}")

print("\n=== Variance check (the same compression pattern found in MUStARD?) ===")
for dim in ["arousal", "valence", "dominance"]:
    print(f"{dim:<10} audio_std0={g0[f'audio_{dim}_z'].std():.3f} audio_std1={g1[f'audio_{dim}_z'].std():.3f}  "
          f"text_std0={g0[f'text_{dim}_z'].std():.3f} text_std1={g1[f'text_{dim}_z'].std():.3f}")
