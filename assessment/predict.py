"""Predict a continuous PHQ-8 total from the aligned turns of one interview."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .model import TrimodalRegressor, TurnRegressor, collate, normalize


def load_features(path):
    with np.load(path, allow_pickle=False) as arrays:
        text = torch.from_numpy(arrays["text"]).float()
        if text.ndim != 2 or text.shape[1] != 768 or not len(text) or not torch.isfinite(text).all():
            raise ValueError("Expected finite text features with shape (turns, 768)")
        row = {"text": text}
        for key, width in (("audio", 158), ("visual", 76)):
            values = torch.from_numpy(arrays[key]).float()
            valid = torch.from_numpy(arrays[key + "_valid"])
            if valid.dtype != torch.bool or valid.shape != values.shape or values.shape != (len(text), width):
                raise ValueError(f"Invalid {key} dimensions or validity mask")
            if not torch.isfinite(values[valid]).all():
                raise ValueError(f"Non-finite observed {key} features")
            row[key] = values.masked_fill(~valid, 0)
            row[key + "_valid"] = valid
    return row


def load_models(checkpoint, text_checkpoint, device):
    meta = torch.load(checkpoint, map_location="cpu", weights_only=True)
    base = torch.load(text_checkpoint, map_location="cpu", weights_only=True)
    if meta["architecture"]["text_architecture"] != base["architecture"]:
        raise ValueError("The text and multimodal checkpoint architectures differ")
    for key in ("mean", "std"):
        if not torch.equal(meta["text_normalization"][key], base["normalization"][key]):
            raise ValueError("Use the text checkpoint paired with this multimodal model")
    model = TrimodalRegressor(**meta["architecture"]).to(device).eval()
    model.load_state_dict(meta["state"])
    text_model = TurnRegressor(**base["architecture"]).to(device).eval()
    text_model.load_state_dict(base["state"])
    return model, text_model, meta


@torch.inference_mode()
def predict(rows, model, text_model, meta, device):
    prepared = normalize(rows, meta["normalization"], meta["text_normalization"])
    inputs = collate(prepared, device)
    scores = model(*inputs)
    missing = torch.tensor([not row["has_av"] for row in prepared], device=device)
    if missing.any():
        scores[missing] = text_model(inputs[0][missing], inputs[1][missing])
    if not torch.isfinite(scores).all():
        raise ValueError("Non-finite regression output")
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    models = load_models(args.checkpoint, args.text_checkpoint, args.device)
    score = predict([load_features(args.features)], *models, args.device).item()
    result = {"phq8_raw": score, "assessment": {"scale_name": "PHQ-8", "origin": "model_estimate",
                                                "total_score": min(24.0, max(0.0, score))}}
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
