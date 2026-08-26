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

### Phase 4 — Is "sarcasm" one thing, or two? (affiliative vs. aggressive irony)

Hypothesis: MUStARD's binary `sarcasm` label conflates two pragmatically different acts — biting/critical irony
("bite") and affiliative/playful irony ("humor") — and these might have distinct acoustic signatures.

- First tried a lexicon shortcut (VADER sentiment on the utterance, then on the context-vs-utterance sentiment
  *gap*, as an irony-strength proxy). Both failed for the same reason: word-level sentiment can't tell "negative
  words because it's a genuine insult" from "negative words because it's an absurd, dark joke" (e.g. "Joey ate my
  last stick of gum, so I killed him" scored as the *most negative* line in the dataset, despite being a joke).
- Abandoned the lexicon and manually judged all 345 sarcastic utterances (utterance + context) on two axes —
  direct↔ironic and friendly↔hostile — producing 4 quadrants: praise, criticism, **bite** (ironic+hostile),
  **humor** (ironic+friendly). See [`quadrant_labels.py`](quadrant_labels.py).
- Result ([`analyze_quadrants.py`](analyze_quadrants.py)): bite is 76% BBT, humor is spread across shows. Pooled,
  bite looked far more dissonant than humor (p=0.0048 for `dissonance_mean`) — but that difference **completely
  vanished after controlling for show** (ANCOVA p=0.91). It was the BBT-vs-other-shows confound in disguise, not
  a bite/humor difference. None of the 12 features tested survived show control.
- Honest takeaway: with this sample size (66 vs. 121, near-total confounding with show) we can't conclude the
  affiliative/aggressive distinction is acoustically real *or* that it isn't — the design can't separate the two
  cleanly. What we can say is it isn't needed for the coarser sarcastic-vs-not detection task, where the Phase
  1-3 markers already work regardless of a line's underlying tone.

### Phase 5 — Dimensional emotion (Valence / Arousal / Dominance)

Tried correlating VAD (the standard 3-dimensional emotion model) against every acoustic feature so far.

- **Text-side VAD**: word-level lookup against the NRC-VAD Lexicon (Mohammad 2018, [`data/vad-nrc-lexicon.csv`](data/vad-nrc-lexicon.csv)) averaged per utterance ([`compute_text_vad.py`](compute_text_vad.py)).
- **Result**: essentially zero correlation between text-VAD and any of our ~30 acoustic features, sarcastic or
  not (even the textbook arousal↔F0 relationship: r=0.016, p=0.69). Likely explanation: this is *scripted,
  acted* dialogue — delivery follows comedic timing/character voice, not the literal emotional weight of the
  words, so text-content and vocal-delivery are largely decoupled regardless of sarcasm. This is itself a
  reasonable argument for why MUStARD is multimodal by design.
- **Audio-side VAD, attempt 1 (rejected)**: a hand-built linear composite of features we already had (e.g.
  audio-valence from `hnr_mean` − `dissonance_mean` − jitter − shimmer). Circular by construction — it partly
  re-encodes findings we already had (like the dissonance effect) rather than testing anything new.
- **Audio-side VAD, attempt 2 (real model)**: switched to `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`
  ([`extract_audio_vad.py`](extract_audio_vad.py)), a wav2vec2 model fine-tuned on MSP-Podcast to regress
  arousal/dominance/valence directly from audio. Needed its own venv (`.venv_vad`) — this torch build only
  supports numpy<2, incompatible with librosa/scipy's numpy>=2 in the main env. Also hit a real bug: the
  checkpoint stores its positional conv embedding with old-style `weight_norm` params (`weight_g`/`weight_v`),
  but this torch version's parametrization expects different internal names, so `from_pretrained` was silently
  leaving that layer randomly initialized. Fixed by reconstructing the true weight from the checkpoint via
  `torch._weight_norm(v, g, dim=2)` and injecting it directly — verified against a silence input (near-neutral
  output) and real utterances (sensible, varied output) before trusting it.
- **Result ([`analyze_audio_vad.py`](analyze_audio_vad.py))**: with the real model, sarcastic utterances score
  significantly **higher** on all three dimensions — arousal, dominance, *and* valence (all p<0.0001, survives
  show-control). The valence direction is counterintuitive until you separate constructs: this model was trained
  to recognize expressive, animated, confident-sounding delivery as positive-valence, which is exactly how a
  well-delivered punchline sounds regardless of how cutting its content is — this doesn't contradict the earlier
  dissonance/HNR finding (voice roughness), it's a different axis (holistic expressiveness vs. timbre).
- **Cross-modal gap** (does audio-VAD vs. text-VAD *mismatch* predict sarcasm, i.e. an irony-detector via
  modality disagreement?): gap is smaller, not larger, for sarcastic lines (p<0.0001) — opposite of the
  hypothesis. But a variance check shows sarcastic utterances have systematically lower variance in *both*
  audio-VAD and text-VAD (crafted punchlines are more uniform than average dialogue), which alone would shrink
  the raw gap mechanically. This held even with the real audio model, so the finding is unresolved, not
  refuted — testing it properly needs a gap metric that isn't confounded by dispersion differences.

### Final model (v1: acoustic + V/A/D)

23 show-normalized features → `StandardScaler` + RBF-`SVC` (`C=10`, `gamma="scale"`), evaluated LOSO
([`train_final_model.py`](train_final_model.py)):

| Group | Features |
|---|---|
| Harmony/prosody (13) | `f0_mean/std/range`, `jitter_local`, `shimmer_local`, `hnr_mean`, `voiced_ratio`, `speaking_rate_proxy`, `energy_std`, `dissonance_mean/std`, `inharmonicity_mean/std` |
| Melody (4) | `melodic_path_length`, `direction_change_rate`, `final_initial_diff`, `num_notes_per_sec` |
| Rhythm (3) | `stress_contrast`, `ioi_cv`, `rpvi_pause_filtered` |
| Audio V/A/D, wav2vec2 (3) | `audio_arousal_wav2vec`, `audio_dominance_wav2vec`, `audio_valence_wav2vec` |

**Balanced (BBT+FRIENDS) show-independent F1 = 0.607**, up from 0.494 (unnormalized) → 0.583 (harmony only) →
0.591 (+ melody) → 0.603 (+ rhythm) → 0.607 (+ audio V/A/D) over the course of this exploration. The V/A/D
addition is the smallest single increment (+0.004) — most of its signal overlaps with features already in the
model (arousal with `energy_std`/`speaking_rate_proxy`, dominance with `f0_mean`, valence with `hnr_mean`/`dissonance_mean`), consistent with it being a real but largely redundant confirmation rather than new information.

### Phase 6 — Does any of this survive leaving MUStARD entirely? (PodSarc)

> **Correction (Phase 10):** everything in this section was run on a 2,885-utterance subset that turned out to
> be exactly the cases GPT-4o and Llama3 *disagreed* on — not a representative sample of PodSarc. See
> [Phase 10](#phase-10--the-podsarc-hard-subset-discovery) for the full
> 11,021-utterance re-run; the qualitative "does the model transfer" conclusions below still hold, but the
> absolute F1 numbers were measured on PodSarc's *hardest* cases and understate performance on the full corpus.

The real test: **PodSarc** (podcast sarcasm, ~11k utterances from *Overly Sarcastic Productions*, GPT-4o +
Llama3 sarcasm labels with human arbitration on disagreements — 2,885 human-verified utterances used here).
Completely different domain: natural conversational co-hosts, not scripted TV performance; clips average
9.1s vs. MUStARD's 4.3s.

**Direct transfer ([`validate_on_podsarc.py`](validate_on_podsarc.py))**: train the v1 model on all of MUStARD,
predict on PodSarc (its own corpus-level z-score in place of show-normalization). **F1 = 0.505 — chance.**
(MUStARD-internal reference: 0.648 random CV, 0.607 LOSO.)

**Why, feature by feature ([`analyze_podsarc_replication.py`](analyze_podsarc_replication.py))**: tested all 33
features (not just MUStARD's chosen 20) for direction-replication, pooled and episode-controlled (PodSarc's
analogue of show-control; its 30 episodes share mostly the same 2-3 hosts, so this tests topic- not
speaker-generalization). Mixed and revealing:

| Result | Features |
|---|---|
| **Direction flips** | `dissonance_mean` (MUStARD's single strongest signal), `hnr_mean`, `f0_mean`, `melodic_path_length`, `direction_change_rate` |
| **Direction replicates** | `voiced_ratio`, `speaking_rate_proxy`, `energy_std`, `inharmonicity_mean`, `stress_contrast`, `ioi_cv`, `rpvi_pause`/`rpvi_pause_filtered` |
| **MUStARD-rejected features recovered here** | `wholetone_fit`, `tonal_clarity`, `ioi_npvi`, `pitch_stress_contrast` (all survive episode-control; scale-fit features that failed completely in MUStARD) |
| **MUStARD-rejected, rejected again** | `major_fit`, `minor_fit`, `majorness`, `mean_interval_size`, `interval_std`, `direction_entropy`, `npvi_voiced` |

Dissonance flipping is why direct transfer collapsed: the model's most heavily-weighted feature actively points
the wrong way in this domain. Rhythm features were the most portable; melody was the least (see below). Real
audio-VAD (wav2vec2) also stopped replicating: none of arousal/dominance/valence survive episode-control here
([`analyze_podsarc_vad_gap.py`](analyze_podsarc_vad_gap.py)), vs. p<0.0001 on all three in MUStARD.

**Melody specifically didn't transfer**: none of MUStARD's 4 kept melody features hold up cleanly —
`final_initial_diff` (the "sarcastic rise" story) goes from MUStARD's headline finding to p=0.93 in PodSarc;
`direction_change_rate` is significant but its *sign flips*; `num_notes_per_sec` and `melodic_path_length` don't
survive episode-control. The exaggerated, theatrically-performed pitch contour that TV sitcom delivery uses for
comedic emphasis doesn't seem to be a thing natural podcast conversation does.

**Retraining from scratch on PodSarc** ([`train_podsarc_classifier.py`](train_podsarc_classifier.py), GroupKFold
by episode): MUStARD's 20 features get F1=0.538 (episode-normalized) when refit on PodSarc's own distribution —
real signal, well above the ~0.38 majority-baseline, but far short of MUStARD-internal performance. A feature
set built *from PodSarc's own replication test* (15 features, including the 4 recovered scale/pitch ones) does
slightly better: **F1=0.545** — the best result on PodSarc, and once again "all 33 features" (0.535) loses to
a validated subset, the same more-features-hurts pattern as every earlier phase.

**Cross-modal VAD gap replicates**: the same "smaller gap for sarcasm" direction and the same variance-compression
confound as MUStARD (`gap_total` p=0.0001 after episode-control; sarcastic-utterance variance is lower in both
audio- and text-VAD here too). Seeing the identical pattern in two structurally unrelated corpora makes this
more likely a genuine cross-domain regularity (crafted punchlines are more uniformly delivered) than a
corpus-specific artifact — but it still doesn't resolve into a validated "mismatch predicts sarcasm" mechanism.

**Sarcasm quadrant analysis needed a 3rd axis here**: PodSarc has no MUStARD-style context field, so the 345-item
manual judging method (direct/ironic × friendly/hostile) was repeated on PodSarc's 287 sarcastic utterances from
text alone. Result: 77% landed in Q1 ("direct", non-hostile) — a degenerate distribution (Q3 "bite" = 1 item,
literally too small to test) that turned out to be a modeling gap, not a property of the data: much of that Q1
mass was **direct but hyperbolic** exaggeration ("Operation Fish Hemsworth is a go", stacked superlatives, mock-
epic language) — sarcasm via overstatement rather than via literal-meaning-flip, which the 2-axis scheme had no
slot for. Adding a third axis, **proportionate ↔ hyperbolic** ([`podsarc_quadrant_labels.py`](podsarc_quadrant_labels.py),
[`analyze_podsarc_octants.py`](analyze_podsarc_octants.py)), split it cleanly: 180 direct+friendly+proportionate
vs. 39 direct+friendly+**hyperbolic** — confirming the missing category was real. But none of 16 acoustic
features tested distinguish hyperbolic from proportionate delivery (all p>0.1, episode-controlled). Plausible
reading: irony functionally *needs* a vocal cue to signal "don't take this literally" (which is exactly what
dissonance/pitch/rhythm were picking up on), while hyperbole is usually already legible from word choice alone
("a million times") and doesn't need one — a genuine negative result that sharpens *why* irony is acoustically
marked and hyperbole isn't, rather than a failure to find something that should have been there.

### Phase 7 — Generic spectral features: do MFCC and mel-spectrograms help?

Deliberately deferred CNN/deep-learning approaches; instead tested classical summary-statistic MFCC (28-dim) and
mel-spectrogram (80-dim, 40 bands, mean+std per band) features (`extract_mfcc_features.py`,
`extract_melspec_features.py`) as an addition to each corpus's already-validated feature set
(`train_with_mfcc.py`, `train_with_spectral.py`):

| Addition | MUStARD (baseline F1=0.607, LOSO) | PodSarc (baseline F1=0.545, GroupKFold) |
|---|---|---|
| + MFCC (28) | 0.556 | 0.559 |
| + mel-spectrogram (80) | 0.587 | 0.559 |
| + both | 0.550 | 0.551 |

Generic spectral features hurt MUStARD and help PodSarc a little — but never as much as, and redundantly with,
the curated feature set (combining MFCC+mel-spec together is *worse* than either alone in both corpora, since
they overlap heavily as descriptions of the same spectral envelope).

**Cosine-similarity check** (`analyze_mfcc_similarity.py`): raw MFCC vectors are nearly identical between
sarcastic and non-sarcastic classes in both corpora (separability gap ≈0.0001 — dominated by speaker/recording
identity, not sarcasm). Per-show/episode normalization flips the centroid correlation to -1.0 but the
separability gap stays tiny (0.0235 MUStARD, 0.0121 PodSarc) — confirming there's no meaningful class structure
in raw spectral-envelope space, consistent with the classifier result above.

### Phase 8 — Formants and voice quality (F1/F2/F3, CPP, H1-H2 spectral tilt)

Two previously-unexplored acoustic dimensions, added together (`harmonic_features.py`'s `formant_features()` and
`voice_quality_features()`): **formants** (F1/F2/F3 mean+std via Praat, plus formant dispersion — articulatory
"openness"/vocal-tract-length proxy) and **voice quality** (Cepstral Peak Prominence for periodicity strength,
H1-H2 spectral tilt for breathy-vs-pressed phonation).

Sanity-checked against synthesized signals before trusting on real data: a source-filter synthesized vowel
(target F1/F2/F3 = 700/1220/2600 Hz) recovered 730/1216/2088 Hz; a 14-harmonic periodic tone gave CPP=0.378 vs.
0.056 for white noise (the first attempt, with only 2 harmonics, failed to separate — cepstral peaks need a
richer harmonic stack to show up); a breathy (1/h^2.5 harmonic falloff) vs. pressed (1/h) synthetic voice gave
H1-H2 = 13.63 dB vs. 4.60 dB.

**Statistical validation** (`analyze_formant_voice_quality.py`, pooled Mann-Whitney + show/episode-ANCOVA, run
independently per corpus — PodSarc's screen does not reuse MUStARD's selections):

| Survives group-control | MUStARD (of 11) | PodSarc-hard-subset (of 11) |
|---|---|---|
| | `f1_mean`, `f2_mean`, `f3_mean`, `f3_std`, `formant_dispersion`, `h1h2_mean`, `h1h2_std` (7) | `f1_mean`, `f3_mean`, `cpp_mean`, `cpp_std`, `h1h2_std` (5) |

`f1_mean`, `f3_mean`, and the H1-H2 tilt validate **independently in both corpora** — the first cross-corpus-
replicated feature family found outside the original 20/15-feature sets.

**Classifier impact — the pattern repeats a 5th time**: despite being genuinely significant, adding these
features (validated-only, or all 11 regardless of significance) reduces held-out F1 in both corpora:

| | baseline | + validated only | + all 11 |
|---|---|---|---|
| MUStARD (LOSO) | 0.607 | 0.584 (7 feats) | 0.590 |
| PodSarc-hard (GroupKFold) | 0.545 | 0.536 (5 feats) | 0.537 |

Statistical significance and multivariate classifier value are, once again, different questions — this makes it
five separate feature families (MFCC, mel-spectrogram, the full 33-feature PodSarc set, and now formants/voice
quality) where "passes the group-controlled test" did not translate into "improves the SVM."

### Phase 9 — Ellipse/QDA regions in V-A-D space (Han & Cha 2017)

A Korean paper on extending Russell's Circumplex Model represents emotion categories as bivariate-Gaussian
confidence ellipses in Valence-Arousal space with a Bayesian/QDA decision boundary between them
(`analyze_vad_ellipse.py`: `ellipse_params()` follows the paper's rotation-from-covariance / semi-axes-from-std
construction exactly; `evaluate_qda_lda()` fits sklearn's QDA/LDA — mathematically the same generative
classifier — under the same show/episode-grouped held-out protocol used throughout, not the paper's own
in-sample accuracy figure). Applied as an independent, generative-model cross-check of the discriminative
ANCOVA conclusions, using the existing wav2vec2 V-A-D features.

**Sarcastic vs. non-sarcastic**: MUStARD LDA F1=0.630, QDA F1=0.567 (QDA in-sample, paper-style: 0.661);
PodSarc(700-sample) LDA F1=0.468, QDA F1=0.516 (in-sample 0.607) — no improvement over the acoustic-feature
SVMs, and the ellipses visibly overlap in both corpora (`data/mustard_vad_ellipse.png`, `data/podsarc_vad_ellipse.png`).

**Sarcasm-subtype ellipses** (do quadrants/octants occupy separable V-A-D regions?): required extending
MUStARD's 2-axis judgments to a 3rd (proportionate/hyperbolic) axis, mirroring what PodSarc needed in Phase 6
— all 345 sarcastic utterances re-judged and cross-checked against the existing 2-axis labels with zero
mismatches (`mustard_quadrant_labels.py`). Every axis/corpus combination shows the same near-total overlap:

| Comparison | n | Centroid distances (Euclidean, V-A-D) |
|---|---|---|
| MUStARD Q1-4 (2-axis) | 345 | 0.009 – 0.047 |
| MUStARD octants (3-axis, new) | 345 (287 plottable) | 0.011 – 0.083 |
| PodSarc octants (3-axis) | 287 (247 plottable) | 0.029 – 0.062 |
| PodSarc Q1-4 (collapsed to 2-axis) | 287 (276 plottable) | 0.037 |

Two structurally different methods (discriminative ANCOVA, generative ellipse/QDA) now agree, in both corpora,
at both axis granularities: fine-grained sarcasm subtypes are not acoustically separable in V-A-D space. That
convergence across independent methods is what makes this a trustworthy negative result rather than one
technique's blind spot.

### Phase 10 — The PodSarc hard-subset discovery

While preparing to test more feature combinations, a direct question exposed a problem with every PodSarc
result up to this point: **how much of PodSarc were we actually using?** The `llms_annotations_with_human_check.json`
annotation file covers all 11,024 utterances across the same 30 episodes used throughout — but
`data/podsarc_labels.csv` (the file every PodSarc script reads) has only 2,885 rows.

Checking the selection: GPT-4o and Llama3 agree on 8,139/11,024 utterances (73.8%) and disagree on exactly
2,885 — and the 2,885 in our dataset are precisely that disagreement set, human-arbitrated (`human_check`
field). **Every PodSarc result in Phases 6-9 was measured on the hardest 26% of the corpus** — the cases where
two different LLMs, given the same text, actively disagreed — not a representative sample.

Extracted the remaining 8,139 "LLM-agreed" utterances: converted mp3→wav (`convert_podsarc_agreed_audio.py`,
via the bundled `imageio_ffmpeg` binary, matching the existing 22050Hz mono format) and ran the full 45-feature
harmonic+formant+voice-quality extraction (checkpointed, ~2.5 hours, 8,139/8,139 succeeded). Combined with the
existing hard subset into a full 11,021-utterance corpus (`data/podsarc_harmonic_features_full.csv`, 30
episodes, 36.5% sarcastic).

**Classifier re-evaluation, full corpus vs. hard-subset-only (GroupKFold by episode):**

| Feature set | Hard subset only (n=2,882) | Full corpus (n=11,021) |
|---|---|---|
| MUStARD-20 | 0.538 | **0.621** |
| PodSarc-validated-15 | 0.541 | **0.611** |
| All 44 features | 0.533 | **0.622** |

PodSarc's F1 ceiling (~0.54, throughout Phases 6-9) was not a fundamental acoustic-feature limitation — it was
an artifact of evaluating exclusively on the cases two different LLMs couldn't agree on. On the full corpus,
harmonic/melody/rhythm/formant features perform comparably to (marginally above) MUStARD's own LOSO F1=0.607.

Also notable: on the full corpus, "all 44 features" (0.622) very slightly *beats* the curated 20-feature set
(0.621) — the "more features hurt" pattern from Phases 1-9 weakens with an order of magnitude more training
data, suggesting it was partly a small-sample overfitting artifact specific to the ~2,900-item hard subset, not
a universal law about these features.

A full episode-ANCOVA re-screen on the 11,021-item corpus found 39/44 features individually significant after
episode-control — plausible given the much larger n makes small effect sizes detectable, so the classifier
result above (not the raw significance count) is the trustworthy part of this finding.

### Phase 11 — Fixing glissando contamination (scale-fit, then jitter/shimmer)

Speech pitch glides continuously between targets rather than landing on discrete notes like an instrument does
— exactly the caveat noted since Phase 1 for the major/minor key-fit features. This phase asks concretely: can
that be fixed, and does anything else in the feature set have the same problem?

**Fix: exclude glide frames from the pitch-class histogram.** Added `scale_features_stable()`
(`harmonic_features.py`) — estimates local pitch velocity (semitones/sec between consecutive voiced frames,
reusing the same 8 semitones/sec threshold `segment_notes()` already used for note-segmentation) and excludes
samples above it before building the histogram major/minor/wholetone-fit correlate against. Sanity-checked with
a synthesized C-major arpeggio whose notes are diluted by fast chromatic "excursion" glides between them: the
naive histogram's `tonal_clarity` was 0.174, the glide-excluded version recovered 0.587 (target notes are
genuinely there; the glides were just burying them).

**Auditing the rest of the feature set for the same problem**: `dissonance`/`inharmonicity` compute from a short
40ms FFT window per frame — a synthetic test (steady tone + one fast glide covering 14% of duration) moved their
means by <2%, so short-window spectral features aren't meaningfully affected. `segment_notes` /
`melodic_interval_features` already had this exact velocity-threshold guard built in from the start (that's
where the 8 semitones/sec threshold reused above actually came from). But the same synthetic test found
**`jitter_local`/`shimmer_local` inflate ~11x/~9x** from that one glide alone — Praat's own period-ratio guard
(excludes pairs >1.3x) doesn't fully protect them. Added `jitter_shimmer_stable_features()`: restricts the same
Praat "Get jitter/shimmer (local)" calls to only the glide-excluded stable time ranges, duration-weighted across
segments. The same synthetic test confirms the fix: stable jitter stays at 0.00031 with or without the glide,
vs. naive jitter's 0.00032 → 0.00356 jump.

**Real-data validation, both fixes, independently in both corpora:**

| | MUStARD (LOSO) | PodSarc-hard (GroupKFold) | PodSarc-full (GroupKFold) |
|---|---|---|---|
| baseline | 0.607 | 0.545 | 0.611 – 0.622 |
| + 6 stable-scale features | **0.613** | 0.549 (noise) | no change |
| + 2 stable-jitter/shimmer (added, not swapped) | **0.617** (new best) | 0.549 (noise) | no change |

MUStARD is where both fixes pay off — `tonal_clarity_stable` is a much cleaner statistical signal than the
naive version (ANCOVA p=0.0005 → p<0.0001), and the previously-rejected naive scale features (which *hurt* when
added, F1=0.595) now help once glide is excluded, confirming the mechanism fixed the actual problem rather than
just adding noise. Swapping jitter/shimmer for their stable versions (instead of adding both) makes MUStARD
*worse* (0.598) — the glide-contaminated original apparently still carried some real information the classifier
was using, so the fix is additive value, not a strict correction. PodSarc shows the same statistical improvement
pattern (`major_fit_stable` becomes significant, p=0.75→0.044) but no classifier movement in either subset —
consistent with the recurring finding that a feature's group-controlled significance and its multivariate
classifier value are separate questions, and that which features matter is domain-specific.

### Lessons that generalized across the whole exploration

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
6. **Lexicon-based text sentiment cannot separate "dark joke" from "genuine insult"** — both look equally
   negative to a word-counter. Judging tone required actually reading each line in context; there's no
   dictionary shortcut for it.
7. **A statistically strong group difference (real audio-VAD, p<0.0001) still only added +0.004 F1** once
   combined with everything else — significance and multivariate marginal value are genuinely different
   questions, confirmed again with a properly validated external model, not just our own hand-built features.
8. When two independent measurements move together (the audio/text VAD gap shrinking for sarcasm), check
   variance before concluding they *agree* — shrinking dispersion in both measurements can mimic convergence.
9. **A model that works within one corpus can fail completely on another (F1 0.607 → 0.505) even when both are
   "the same task."** Show-independence inside MUStARD and domain-independence across MUStARD/PodSarc are
   different, unrelated tests — passing the first proves nothing about the second.
10. **Feature significance is domain-specific, not just show-specific.** MUStARD's strongest single feature
    (dissonance) flipped sign in PodSarc, while several features MUStARD rejected outright turned out to matter
    there. A feature set earns its place per-domain; it doesn't transfer by authority.
11. **Not every sarcasm marker needs a matching acoustic marker.** Irony (literal-meaning flip) is acoustically
    loud because the mismatch has to be signaled somehow; hyperbole (scale exaggeration) can be entirely lexical
    and still read as sarcastic. A clean null result on the hyperbole axis is informative, not a dead end.
12. **Statistical significance and classifier value kept being different questions, five separate times**
    (MFCC, mel-spectrogram, the full 33-feature PodSarc set, formants, voice quality). A feature surviving
    group-controlled ANCOVA is necessary but nowhere near sufficient evidence it belongs in the model.
13. **Always check what a filtered dataset was actually filtered by.** PodSarc's 2,885-row label file looked
    like "the corpus"; it was actually the 26% GPT-4o and Llama3 disagreed on. Every PodSarc conclusion in
    Phases 6-9 was true, but true about the hardest slice — always confirm a dataset is what it claims to be
    before trusting conclusions drawn from it, especially when a filename says "human-verified" without saying
    verified *how* or *which subset*.
14. **A small-sample pattern can be an artifact of the sample size, not the features.** "More features hurt"
    held reliably from Phase 1 through Phase 9 (~700-2,900 items) and then nearly vanished at 11,021 items —
    a conclusion that looked like a stable law was partly overfitting risk scaling with n.
15. **A named caveat is a lead, not just a disclaimer.** "Speech pitch glides continuously" was written down as
    a limitation back in Phase 1 and left alone for ten phases. Actually acting on it found a real fix (scale-fit)
    and a previously-unknown problem in a feature used since Phase 1 (jitter/shimmer, 11x inflated by one glide).
16. **A theoretically-correct fix doesn't have to replace the thing it fixes.** Swapping glide-contaminated
    jitter/shimmer for the corrected version made MUStARD worse; adding the correction *alongside* the original
    gave the best result of the whole project. The contaminated signal still carried real information.

### Files

| File | Purpose |
|---|---|
| `harmonic_features.py` | All acoustic feature extraction (dissonance, prosody, scale-fit, melody, rhythm) + sanity checks (`python harmonic_features.py`) |
| `extract_harmonic_features.py` | Batch CLI: audio dir → `data/harmonic_features.csv` |
| `extract_mfcc_features.py` | MFCC baseline extraction (used in the rejected MFCC-combination experiment) |
| `train_harmonic_svm.py` | Baseline SVM: random-CV vs. single-split show-independent eval |
| `train_harmonic_svm_normalized.py` | Adds show/speaker normalization |
| `train_combined_svm.py` | Harmony+MFCC combination + permutation importance |
| `quadrant_labels.py` | Manual direct/ironic × friendly/hostile judgments for all 345 sarcastic utterances |
| `analyze_quadrants.py` | Phase 4: bite-vs-humor acoustic comparison (pooled + show-controlled) |
| `compute_text_vad.py` | Text-side V/A/D via the NRC-VAD Lexicon |
| `extract_audio_vad.py` | Audio-side V/A/D via wav2vec2 (includes the weight_norm checkpoint fix) — run in `.venv_vad` |
| `analyze_audio_vad.py` | Phase 5: sarcasm-vs-VAD tests + the cross-modal gap analysis |
| `train_final_model.py` | Final 23-feature model, LOSO-evaluated |
| `data/harmonic_features.csv` | Harmony/melody/rhythm feature table (690 rows) with labels |
| `data/audio_vad.csv` | wav2vec2 audio V/A/D per utterance |
| `data/vad-nrc-lexicon.csv` | NRC-VAD Lexicon (Mohammad 2018), used by `compute_text_vad.py` |
| `extract_podsarc_features.py` | Checkpointed batch CLI: harmony/melody/rhythm for PodSarc (resumable — writes every N utterances) |
| `extract_podsarc_audio_vad.py` | Checkpointed wav2vec2 audio V/A/D for PodSarc — run in `.venv_vad` |
| `compute_podsarc_text_vad.py` | Text-side V/A/D for PodSarc via the NRC-VAD Lexicon |
| `validate_on_podsarc.py` | Direct transfer test: MUStARD-trained v1 model evaluated on PodSarc |
| `analyze_podsarc_replication.py` | Does each MUStARD feature's direction/significance replicate in PodSarc? (all 33, pooled + episode-ANCOVA) |
| `train_podsarc_classifier.py` | Retrains from scratch on PodSarc, GroupKFold by episode; compares MUStARD's 20 vs. a PodSarc-validated 15 vs. all 33 |
| `analyze_podsarc_vad_gap.py` | PodSarc replication of the sarcasm-vs-VAD test and the cross-modal gap analysis |
| `podsarc_quadrant_labels.py` | Manual 3-axis (direct/ironic × friendly/hostile × proportionate/hyperbolic) judgments for PodSarc's 287 sarcastic utterances |
| `analyze_podsarc_octants.py` | 8-octant breakdown + proportionate-vs-hyperbolic acoustic test |
| `data/podsarc_labels.csv` | **The GPT-4o/Llama3 disagreement subset only** (2,885 rows, human-arbitrated) — not the full corpus, see Phase 10 |
| `data/podsarc_labels_sample.csv` | 700-item stratified subsample (of the disagreement subset) used for the (slower) wav2vec2 VAD pass |
| `data/podsarc_harmonic_features.csv` | Harmony/melody/rhythm/formant/voice-quality features (45 cols) for the 2,885-item disagreement subset |
| `data/podsarc_audio_vad.csv` | wav2vec2 audio V/A/D for the 700-item PodSarc sample |
| `extract_melspec_features.py` | Mel-spectrogram summary-stat extraction (80-dim, 40 bands mean+std) |
| `analyze_mfcc_similarity.py` | Cosine-similarity (centroid + intra/inter-class) check for MFCC vectors, raw vs. group-normalized — `--corpus mustard\|podsarc` |
| `train_with_mfcc.py` | Phase 7: adds MFCC to each corpus's validated feature set — `--corpus mustard\|podsarc` |
| `train_with_spectral.py` | Phase 7: adds MFCC, mel-spectrogram, and both together — `--corpus mustard\|podsarc` |
| `data/mfcc_features.csv`, `data/podsarc_mfcc_features.csv` | MFCC feature tables |
| `data/melspec_features.csv`, `data/podsarc_melspec_features.csv` | Mel-spectrogram feature tables |
| `analyze_formant_voice_quality.py` | Phase 8: pooled + show/episode-ANCOVA for formants + CPP/H1-H2, run independently per corpus — `--corpus mustard\|podsarc` |
| `train_with_formants.py`, `train_with_formants_podsarc.py` | Phase 8: adds validated (and all 11) formant/voice-quality features to each corpus's baseline model |
| `analyze_vad_ellipse.py` | Phase 9: ellipse-region + QDA/LDA method (Han & Cha 2017) for sarcastic-vs-non-sarcastic in wav2vec2 V-A-D space — `--corpus mustard\|podsarc` |
| `analyze_vad_ellipse_quadrants.py`, `analyze_vad_ellipse_mustard_octants.py` | Phase 9: MUStARD sarcasm-subtype ellipses, 2-axis and 3-axis |
| `analyze_vad_ellipse_octants.py`, `analyze_vad_ellipse_podsarc_q1q4.py` | Phase 9: PodSarc sarcasm-subtype ellipses, 3-axis and collapsed-2-axis |
| `mustard_quadrant_labels.py` | Manual 3-axis judgments (extends `quadrant_labels.py` with proportionate/hyperbolic) for all 345 MUStARD sarcastic utterances |
| `convert_podsarc_agreed_audio.py` | Phase 10: mp3→wav conversion (via bundled `imageio_ffmpeg`) for the 8,139 LLM-agreed PodSarc utterances |
| `data/podsarc_labels_agreed.csv` | Phase 10: the 8,139 LLM-agreed utterances (sarcasm = the agreed GPT-4o/Llama3 label) |
| `data/podsarc_harmonic_features_agreed.csv` | Phase 10: harmonic/formant/voice-quality features (45 cols) for the 8,139 agreed utterances |
| `data/podsarc_harmonic_features_full.csv` | Phase 10/11: the full 11,021-utterance corpus (disagreement + agreed subsets combined, tagged by `subset`), including the Phase 11 stable-scale and stable-jitter/shimmer columns |
| `extract_stable_scale_only.py` | Phase 11: fast incremental extraction of just the 6 glide-excluded scale features (skips the full 45-feature pipeline) |
| `extract_stable_jitter_shimmer_only.py` | Phase 11: fast incremental extraction of just the glide-excluded jitter/shimmer features |
| `data/podsarc_stable_scale.csv`, `data/podsarc_stable_scale_agreed.csv` | Phase 11: `scale_features_stable()` output for the hard-subset and agreed PodSarc utterances |
| `data/podsarc_stable_jitter_shimmer.csv`, `data/podsarc_stable_jitter_shimmer_agreed.csv` | Phase 11: `jitter_shimmer_stable_features()` output for the hard-subset and agreed PodSarc utterances |
