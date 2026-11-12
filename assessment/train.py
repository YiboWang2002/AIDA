"""Train the text regressor, then fine-tune the multimodal regressor."""

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path

import torch

from .model import TrimodalRegressor, TurnRegressor, collate, fit_normalization, normalize
from .predict import load_features


def read_split(manifest):
    records = {}
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        for entry in csv.DictReader(stream):
            path = (manifest.parent / entry["features"]).resolve()
            score = float(entry["phq8"])
            if path in records or not 0 <= score <= 24:
                raise ValueError("Expected a unique interview and a PHQ-8 total from 0 to 24")
            records[path] = dict(load_features(path), total=score)
    if not records:
        raise ValueError("Empty split")
    return records


def batches(rows, order):
    for start in range(0, len(order), 16):
        yield [rows[i] for i in order[start:start + 16]]


def targets(rows, device):
    return torch.tensor([r["total"] for r in rows], device=device)


def train_text(fit, dev, device):
    means = torch.stack([r["text"].mean(0) for r in fit])
    squares = torch.stack([r["text"].square().mean(0) for r in fit])
    mean = means.mean(0)
    std = (squares.mean(0) - mean.square()).clamp_min(1e-6).sqrt()
    fit = [dict(r, text=(r["text"] - mean) / std) for r in fit]
    dev = [dict(r, text=(r["text"] - mean) / std) for r in dev]
    totals = targets(fit, device)
    arch = dict(embedding_dim=768, hidden_dim=64, dropout=0.2, target_mean=totals.mean().item(),
                target_std=totals.std(unbiased=False).clamp_min(1e-6).item())
    model = TurnRegressor(**arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    for _ in range(200):
        model.train()
        opt.zero_grad(set_to_none=True)
        for batch in batches(fit, torch.randperm(len(fit)).tolist()):
            text, mask = collate(batch, device)[:2]
            errors = (model(text, mask) - targets(batch, device)) / model.target_std
            (errors.square().sum() / len(fit)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        model.eval()
        with torch.inference_mode():
            out = torch.cat([model(*collate(b, device)[:2]) for b in batches(dev, range(len(dev)))])
        yield (out - targets(dev, device)).abs().mean().item(), {
            "architecture": arch, "normalization": {"mean": mean, "std": std}, "state": model.state_dict()}


def train_multimodal(fit, dev, device, text_checkpoint):
    base = torch.load(text_checkpoint, map_location="cpu", weights_only=True)
    original = TurnRegressor(**base["architecture"]).to(device).eval()
    original.load_state_dict(base["state"])
    norm = fit_normalization(fit)
    fit = [r for r in normalize(fit, norm, base["normalization"]) if r["has_av"]]
    dev = normalize(dev, norm, base["normalization"])
    if not fit:
        raise ValueError("No observed acoustic or visual features in the training split")
    arch = {"text_architecture": base["architecture"], "modality_dropout": 0.1}
    model = TrimodalRegressor(**arch).to(device)
    model.text.load_state_dict(base["state"])
    opt = torch.optim.AdamW([{"params": model.new_parameters(), "lr": 3e-4},
                             {"params": model.text.parameters(), "lr": 3e-5}], weight_decay=0.01)
    for epoch in range(100):
        model.configure_training_phase(epoch < 5)
        for batch in batches(fit, torch.randperm(len(fit)).tolist()):
            opt.zero_grad(set_to_none=True)
            scores = model(*collate(batch, device))
            ((scores - targets(batch, device)) / model.text.target_std).square().mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
        model.eval()
        with torch.inference_mode():
            outputs = []
            for batch in batches(dev, range(len(dev))):
                inputs = collate(batch, device)
                scores = model(*inputs)
                missing = torch.tensor([not r["has_av"] for r in batch], device=device)
                if missing.any():
                    scores[missing] = original(inputs[0][missing], inputs[1][missing])
                outputs.append(scores)
        yield (torch.cat(outputs) - targets(dev, device)).abs().mean().item(), {
            "architecture": arch, "state": model.state_dict(), "normalization": norm,
            "text_normalization": {k: base["normalization"][k] for k in ("mean", "std")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("text", "multimodal"), required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-checkpoint", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.stage == "multimodal" and args.text_checkpoint is None:
        parser.error("--stage multimodal requires --text-checkpoint")
    if args.output.exists():
        parser.error("Choose a new output checkpoint")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    train = read_split(args.train)
    dev = read_split(args.dev)
    if train.keys() & dev.keys():
        raise ValueError("Interview file appears in both train and dev")
    torch.manual_seed(args.seed if args.seed is not None else (22 if args.stage == "text" else 42))
    fit, valid = list(train.values()), list(dev.values())
    epochs = (train_text(fit, valid, args.device) if args.stage == "text" else
              train_multimodal(fit, valid, args.device, args.text_checkpoint))
    best, stale = math.inf, 0
    for epoch, (mae, checkpoint) in enumerate(epochs, start=1):
        if not math.isfinite(mae):
            raise ValueError("Non-finite validation MAE")
        print(f"Epoch {epoch:3d}  validation MAE {mae:.4f}")
        if mae < best:
            with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
                candidate = Path(temp) / "checkpoint.pt"
                torch.save(checkpoint, candidate)
                if best == math.inf:
                    os.link(candidate, args.output)
                else:
                    candidate.replace(args.output)
            best, stale = mae, 0
        else:
            stale += 1
            if stale == 20:
                break


if __name__ == "__main__":
    main()
