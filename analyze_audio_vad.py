#!/usr/bin/env python
"""Statistical comparison of wav2vec2-derived audio V/A/D between sarcastic and
non-sarcastic utterances (pooled, within-show, and show-controlled ANCOVA),
plus a re-run of the cross-modal text-vs-audio VAD gap analysis using this
real model instead of the earlier hand-built linear-composite proxy.
"""
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import numpy as np

audio_vad = pd.read_csv("data/audio_vad.csv")
text_vad = pd.read_csv("data/vad_scores.csv")
features = pd.read_csv("data/harmonic_features.csv")

df = audio_vad.merge(features[["id", "sarcasm", "show"]], on="id", how="inner")
df = df.merge(text_vad, on="id", how="left")
print("n =", len(df))

audio_dims = ["audio_arousal_wav2vec", "audio_dominance_wav2vec", "audio_valence_wav2vec"]

print("\n=== Pooled Mann-Whitney: sarcasm=1 vs 0 (real audio VAD) ===")
g0, g1 = df[df.sarcasm == 0], df[df.sarcasm == 1]
for c in audio_dims:
    u, p = stats.mannwhitneyu(g0[c].dropna(), g1[c].dropna())
    direction = "up" if g1[c].median() > g0[c].median() else "down"
    print(f"{c:<26} median0={g0[c].median():.3f} median1={g1[c].median():.3f}  "
          f"sarcastic {direction}  p={p:.4f}")

print("\n=== Within-show (BBT, FRIENDS) ===")
for show in ["BBT", "FRIENDS"]:
    sub = df[df.show == show]
    s0, s1 = sub[sub.sarcasm == 0], sub[sub.sarcasm == 1]
    print(f"--- {show} (n0={len(s0)}, n1={len(s1)}) ---")
    for c in audio_dims:
        u, p = stats.mannwhitneyu(s0[c].dropna(), s1[c].dropna())
        print(f"  {c:<26} p={p:.4f}")

print("\n=== ANCOVA (show-controlled): dim ~ sarcasm + C(show) ===")
for c in audio_dims:
    model = smf.ols(f"{c} ~ sarcasm + C(show)", data=df).fit()
    print(f"{c:<26} sarcasm coef={model.params['sarcasm']:>9.4f}  p={model.pvalues['sarcasm']:.4f}")

# --- Re-run the cross-modal gap analysis with the real audio-VAD model ---
print("\n\n=== Cross-modal gap: real wav2vec2 audio-VAD vs NRC-VAD text scores ===")
gap_df = df.dropna(subset=["valence", "arousal", "dominance"]).copy()


def z(s):
    return (s - s.mean()) / s.std()


gap_df["audio_arousal_z"] = z(gap_df.audio_arousal_wav2vec)
gap_df["audio_valence_z"] = z(gap_df.audio_valence_wav2vec)
gap_df["audio_dominance_z"] = z(gap_df.audio_dominance_wav2vec)
gap_df["text_arousal_z"] = z(gap_df.arousal)
gap_df["text_valence_z"] = z(gap_df.valence)
gap_df["text_dominance_z"] = z(gap_df.dominance)

gap_df["gap_arousal"] = (gap_df.audio_arousal_z - gap_df.text_arousal_z).abs()
gap_df["gap_valence"] = (gap_df.audio_valence_z - gap_df.text_valence_z).abs()
gap_df["gap_dominance"] = (gap_df.audio_dominance_z - gap_df.text_dominance_z).abs()
gap_df["gap_total"] = np.sqrt(gap_df.gap_arousal**2 + gap_df.gap_valence**2 + gap_df.gap_dominance**2)

gap_df.to_csv("data/audio_vad_gap.csv", index=False)

g0, g1 = gap_df[gap_df.sarcasm == 0], gap_df[gap_df.sarcasm == 1]
print(f"{'gap':<14}{'median0':>10}{'median1':>10}{'pooled_p':>10}{'ANCOVA_p':>10}")
for c in ["gap_arousal", "gap_valence", "gap_dominance", "gap_total"]:
    u, p_pooled = stats.mannwhitneyu(g0[c].dropna(), g1[c].dropna())
    model = smf.ols(f"{c} ~ sarcasm + C(show)", data=gap_df).fit()
    p_ancova = model.pvalues["sarcasm"]
    print(f"{c:<14}{g0[c].median():>10.3f}{g1[c].median():>10.3f}{p_pooled:>10.4f}{p_ancova:>10.4f}")

# Diagnostic: check for the same variance-compression artifact found with the
# hand-built composite, so we don't misread a real effect this time either.
print("\n=== Variance check (std) by sarcasm status ===")
for dim in ["arousal", "valence", "dominance"]:
    print(f"{dim:<10} audio_std0={g0[f'audio_{dim}_z'].std():.3f} audio_std1={g1[f'audio_{dim}_z'].std():.3f}  "
          f"text_std0={g0[f'text_{dim}_z'].std():.3f} text_std1={g1[f'text_{dim}_z'].std():.3f}")
