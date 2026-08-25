# Sarcasm Detection: An Acoustic / Music-Theoretic Exploration

This repository started as a fork of **MUStARD** (Castro et al., ACL 2019), a multimodal sarcasm-detection
dataset built from *Friends*, *The Big Bang Theory*, *The Golden Girls*, and *Sarcasmaholics Anonymous*. It now
also contains a separate, from-scratch exploration: **can sarcasm be detected from the raw acoustics of speech,
using features borrowed from music theory (harmony, melody, rhythm) rather than the dataset's original
text/BERT/MFCC pipeline?**

## Original dataset (condensed)

690 audiovisual utterances, each labeled `sarcasm: true/false` with speaker, show, and preceding context, in
[`data/sarcasm_data.json`](data/sarcasm_data.json). Raw video clips: [HuggingFace Hub](https://huggingface.co/datasets/MichiganNLP/MUStARD/resolve/main/mmsd_raw_data.zip).
The original text/audio/video SVM baseline is in [`train_svm.py`](train_svm.py) (`python train_svm.py -h`).

```bibtex
@inproceedings{mustard,
    title = "Towards Multimodal Sarcasm Detection (An \_Obviously\_ Perfect Paper)",
    author = "Castro, Santiago and Hazarika, Devamanyu and P{\'e}rez-Rosas, Ver{\'o}nica and
      Zimmermann, Roger and Mihalcea, Rada and Poria, Soujanya",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    year = "2019", publisher = "Association for Computational Linguistics",
}
```

---

## The acoustic/music-theoretic exploration

### Why

Standard sarcasm-detection pipelines lean on text (BERT) or generic spectral features (MFCC). This exploration
instead asks whether concepts from music theory's three elements — **harmony**, **melody**, **rhythm** — applied
to a speaker's voice can carry a sarcasm signal on their own, and whether that signal is interpretable rather
than a black box.

### Setup

- Python 3.12 venv (Python 3.14 was tried first; numba/llvmlite, a librosa dependency, has no wheel for it yet)
- `numpy`, `scipy`, `praat-parselmouth` (Praat bindings — pitch, intensity, jitter/shimmer, HNR), `librosa`
  (MFCC only), `pandas`, `scikit-learn`, `statsmodels`
- Audio: raw clips → `ffmpeg` (via `imageio-ffmpeg`, a static binary — `brew install ffmpeg` tries to rebuild
  half of Homebrew from source on an older macOS and is not worth it) → mono 22.05kHz WAV per utterance

### Methodology (the two decisions that mattered most)

1. **Never trust random 5-fold CV alone.** `sarcasm_data.json`'s `show` field lets the same actor's voice
   appear in both train and test under random CV, so a model can win by memorizing *who* is talking rather than
   detecting sarcasm. Every result below is evaluated **show-independent**: train on 3 shows, test on a show
   with zero shared speakers, and eventually **Leave-One-Show-Out (LOSO)** across all 4 shows, averaged only
   over the two class-balanced shows (BBT, FRIENDS — GoldenGirls and Sarcasmaholics are almost single-class and
   inflate F1 trivially).
2. **Show-normalize every feature** (z-score within each show) before classification. Un-normalized features let
   the model pick up each show's baseline vocal register/recording level instead of the actual within-show
   sarcastic shift — this one change took show-independent F1 from 0.494 (chance level) to 0.583.

A feature only earns a place in the final model if it passes an ANCOVA (`feature ~ sarcasm + C(show)`, i.e.
significant after controlling for show) — single train/test-split permutation importance turned out to be too
noisy at this sample size (690 rows, 4 shows) to trust for feature selection on its own.

### Phase 1 — Harmony

Implemented Sethares' (1993) sensory-dissonance model directly from FFT spectral peaks (no `essentia` — no
macOS/py3.14 wheel), plus inharmonicity, and standard prosody (F0, jitter, shimmer, HNR via Praat).

- Sanity-checked against music theory itself first: synthetic musical intervals ranked correctly (minor 2nd most
  dissonant, octave/unison least) before trusting the implementation on speech.
- **`dissonance_mean`/`dissonance_std` turned out to be the single most robust signal in the whole project** —
  significant (p<0.0001) pooled, within BBT alone, within FRIENDS alone, and after controlling for show.
  Sarcastic utterances have ~3x higher dissonance and lower HNR (rougher voice).
- Tried and **rejected**: per-speaker normalization (worse than per-show — too few samples per actor, and a
  speaker's own baseline absorbed some of their real sarcasm style); combining with MFCC (MFCC is a speaker-ID
  feature by design, it re-introduced the show-identity leak that show-normalization had just fixed, dropping
  show-independent F1 from 0.583 to 0.513); Krumhansl-Schmuckler major/minor/whole-tone key-fit on the pitch
  contour (theoretically appealing but speech pitch glides continuously rather than landing on scale degrees —
  significance didn't survive a single show in isolation, and no version, coarse or microtonal, beat the
  baseline classifier).

### Phase 2 — Melody

Melody = pitch organized in time, distinct from static F0 stats and from harmony's simultaneous-pitch view.
Two views: continuous contour (path length, direction-change rate, slope, start→end pitch difference) and a
discretized "note" sequence via stable-pitch-region segmentation (interval size, direction entropy).

- Sanity-checked with a synthetic 4-note melody (exact interval/duration recovery) — caught and fixed a bug
  where Praat's frame-to-frame pitch jitter was being counted as real melodic direction changes.
- 4 of 8 features survived show-controlled ANCOVA: `melodic_path_length`, `direction_change_rate`,
  `final_initial_diff`, `num_notes_per_sec` (all p<0.01). Direction: sarcastic utterances move **less** overall
  (flatter, fewer direction changes) but end **higher relative to their start** — a flat delivery with a
  characteristic terminal rise.
- Adding all 8 features hurt generalization; adding only the 4 ANCOVA-validated ones was **the first genuine,
  LOSO-confirmed improvement** in the project: balanced (BBT+FRIENDS) show-independent F1 0.570 → 0.591,
  improving on *both* shows independently (not just one lucky split).

### Phase 3 — Rhythm

Rhythm = duration pattern (long-short) + accent pattern (strong-weak). Implemented via the Pairwise Variability
Index (PVI, from the speech-rhythm literature: Grabe & Low 2002) on voiced/pause interval durations, and
intensity-peak-based stress contrast / inter-onset-interval regularity.

- Sanity-checked with synthetic regular vs. irregular+accented beat tracks — caught and fixed a bug where
  "stress contrast" was measuring peak-vs-silence loudness instead of the intended peak-vs-peak contrast.
- `stress_contrast` was the strongest new signal (ANCOVA p<0.0001): sarcastic utterances alternate strong/weak
  loudness **less**, reinforcing the "flatter delivery" story from melody. `ioi_cv` and `rpvi_pause` were
  weaker/borderline.
- Iterated on the metric definitions themselves:
  - `pitch_stress_contrast` (pitch-accent instead of loudness-accent) — not significant, dropped.
  - `ioi_npvi` (outlier-robust local variant of `ioi_cv`) — **worse** than the original. The "outlier" it
    suppresses (utterance-final lengthening) turned out to overlap with the real sarcastic-rise signal from
    Phase 2, so robustness traded away signal along with noise.
  - `rpvi_pause_filtered` (excludes <100ms gaps, i.e. consonant closures, keeping only real pauses) —
    **improved** on the original (ANCOVA p=0.057 → p=0.0034), and was the one rhythm feature that reliably
    helped the classifier.
- Final rhythm set (`stress_contrast`, `ioi_cv`, `rpvi_pause_filtered`) raised balanced F1 0.591 → **0.603**
  (BBT 0.586, FRIENDS 0.621).

### Final model

20 show-normalized features → `StandardScaler` + RBF-`SVC` (`C=10`, `gamma="scale"`), evaluated LOSO:

| Group | Features |
|---|---|
| Harmony/prosody (13) | `f0_mean/std/range`, `jitter_local`, `shimmer_local`, `hnr_mean`, `voiced_ratio`, `speaking_rate_proxy`, `energy_std`, `dissonance_mean/std`, `inharmonicity_mean/std` |
| Melody (4) | `melodic_path_length`, `direction_change_rate`, `final_initial_diff`, `num_notes_per_sec` |
| Rhythm (3) | `stress_contrast`, `ioi_cv`, `rpvi_pause_filtered` |

**Balanced (BBT+FRIENDS) show-independent F1 = 0.603**, up from 0.494 (unnormalized) / 0.583 (harmony-only, show-normalized) at the start of this exploration.

### Lessons that generalized across all three phases

1. **Random CV lies here.** Every "improvement" was re-checked show-independent before being believed; several
   looked like wins under random CV and were actually show-identity leakage.
2. **A single held-out show isn't enough either.** The melody improvement wasn't trusted until it held up under
   full LOSO on both balanced shows independently — Friends alone could have been a lucky split.
3. **More features hurt more often than they helped** — MFCC, full 8-feature melody set, full 5-feature scale
   set, all-8 rhythm set all *reduced* show-independent F1 versus a curated subset. Only ANCOVA-validated,
   selectively-added features ever improved it.
4. **A metric made "more robust" can lose the signal it was meant to protect** (`ioi_npvi` vs `ioi_cv`) — check
   the theoretical fix against real data, not just intuition.
5. Literal music-theory transplants (major/minor key-fit) were the weakest link: they assume a discretized,
   quantized pitch system that continuous speech F0 doesn't actually have. The prosodic/timbral analogues
   (dissonance, HNR) worked; the literal scale-degree framework didn't.

### Files

| File | Purpose |
|---|---|
| `harmonic_features.py` | All acoustic feature extraction (dissonance, prosody, scale-fit, melody, rhythm) + sanity checks (`python harmonic_features.py`) |
| `extract_harmonic_features.py` | Batch CLI: audio dir → `data/harmonic_features.csv` |
| `extract_mfcc_features.py` | MFCC baseline extraction (used in the rejected MFCC-combination experiment) |
| `train_harmonic_svm.py` | Baseline SVM: random-CV vs. single-split show-independent eval |
| `train_harmonic_svm_normalized.py` | Adds show/speaker normalization |
| `train_combined_svm.py` | Harmony+MFCC combination + permutation importance |
| `data/harmonic_features.csv` | Final extracted feature table (690 rows) with labels |
