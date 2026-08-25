#!/usr/bin/env python
"""Extracts real audio-derived arousal/dominance/valence for every MUStARD
utterance using audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
(trained on MSP-Podcast), replacing the earlier hand-built linear-composite
audio-VAD proxy.

Must run in .venv_vad (isolated from the main .venv312 because this model's
torch build only supports numpy<2, while librosa/scipy need numpy>=2).

Works around a checkpoint/torch-version mismatch: the checkpoint stores the
positional conv embedding with old-style weight_norm params (weight_g,
weight_v), but this torch version's weight_norm parametrization expects a
different internal naming, causing Wav2Vec2Model.from_pretrained to silently
leave that layer randomly initialized. We reconstruct the real weight from
the checkpoint's weight_g/weight_v via torch._weight_norm and inject it
directly (verified by comparing to Wav2Vec2's dim=2 weight_norm convention).
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P
import torchaudio
import pandas as pd
from huggingface_hub import hf_hub_download
from transformers import Wav2Vec2Processor, Wav2Vec2PreTrainedModel, Wav2Vec2Model
from tqdm import tqdm

MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"


class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = self.dropout(features)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        hidden_states = self.wav2vec2(input_values)[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        return hidden_states, self.classifier(hidden_states)


def load_model():
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = EmotionModel.from_pretrained(MODEL_NAME)

    conv = model.wav2vec2.encoder.pos_conv_embed.conv
    if P.is_parametrized(conv, "weight"):
        P.remove_parametrizations(conv, "weight", leave_parametrized=True)
    sd = torch.load(hf_hub_download(MODEL_NAME, "pytorch_model.bin"), map_location="cpu")
    with torch.no_grad():
        conv.weight.copy_(torch._weight_norm(
            sd["wav2vec2.encoder.pos_conv_embed.conv.weight_v"],
            sd["wav2vec2.encoder.pos_conv_embed.conv.weight_g"], dim=2))
        conv.bias.copy_(sd["wav2vec2.encoder.pos_conv_embed.conv.bias"])
    model.eval()
    return processor, model


def predict(path: str, processor, model) -> tuple:
    y, sr = torchaudio.load(path)
    if sr != 16000:
        y = torchaudio.functional.resample(y, sr, 16000)
    y = y.mean(dim=0).numpy()
    inputs = processor(y, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        _, logits = model(inputs.input_values)
    arousal, dominance, valence = logits.squeeze().tolist()
    return arousal, dominance, valence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--output", default="data/audio_vad.csv")
    args = parser.parse_args()

    processor, model = load_model()

    rows = []
    for filename in tqdm(sorted(os.listdir(args.audio_dir)), desc="wav2vec2 VAD"):
        if not filename.endswith(".wav"):
            continue
        id_ = filename[:-4]
        try:
            a, d, v = predict(os.path.join(args.audio_dir, filename), processor, model)
        except Exception as e:
            print(f"Failed on {id_}: {e}")
            continue
        rows.append({"id": id_, "audio_arousal_wav2vec": a,
                     "audio_dominance_wav2vec": d, "audio_valence_wav2vec": v})

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
