"""Encode one DAIC-WOZ interview as aligned participant-turn features."""

import argparse
import csv
import io
import math
import zipfile
from itertools import groupby
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from .features import stream_aggregate


def participant_turns(stream):
    rows = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig"), delimiter="\t")
    turns = []
    for speaker, lines in groupby(rows, key=lambda r: r["speaker"]):
        if speaker == "Ellie":
            continue
        if speaker != "Participant":
            raise ValueError(f"Unrecognized speaker: {speaker}")
        utterances, intervals = [], []
        for row in lines:
            if row["value"].strip():
                utterances.append(row["value"])
            start, stop = float(row["start_time"]), float(row["stop_time"])
            if not math.isfinite(start + stop) or stop < start:
                raise ValueError("Invalid transcript interval")
            if stop > max(0, start):
                intervals.append([max(0, start), stop])
        if utterances:
            turns.append({"text": "\n".join(utterances), "intervals": intervals})
    if not turns:
        raise ValueError("No participant speech in transcript")
    return turns


@torch.inference_mode()
def encode_text(turns, encoder, device):
    tokenizer = AutoTokenizer.from_pretrained(encoder)
    model = AutoModel.from_pretrained(encoder, dtype=torch.float32).eval().to(device)
    capacity = 384 - tokenizer.num_special_tokens_to_add(pair=False)
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None or capacity != 382:
        raise ValueError("Expected the all-mpnet-base-v2 tokenizer")
    windows, owners, weights = [], [], []
    for i, turn in enumerate(turns):
        ids = tokenizer.encode(turn["text"], add_special_tokens=False, truncation=False)
        if not ids:
            raise ValueError("Empty tokenized participant turn")
        for start in range(0, len(ids), capacity):
            window = ids[start:start + capacity]
            windows.append([tokenizer.bos_token_id, *window, tokenizer.eos_token_id])
            owners.append(i)
            weights.append(len(window))
    embedding = torch.zeros(len(turns), model.config.hidden_size, device=device)
    for start in range(0, len(windows), 32):
        selected = windows[start:start + 32]
        batch = tokenizer.pad({"input_ids": selected}, padding=True, return_tensors="pt").to(device)
        mask = batch["attention_mask"].unsqueeze(-1)
        hidden = model(**batch).last_hidden_state
        pooled = F.normalize((hidden * mask).sum(1) / mask.sum(1), dim=-1)
        owner = torch.tensor(owners[start:start + len(selected)], device=device)
        weight = torch.tensor(weights[start:start + len(selected)], device=device)
        embedding.index_add_(0, owner, pooled * weight[:, None])
    return F.normalize(embedding, dim=-1).cpu().numpy()


def prepare(interview, encoder, device):
    descriptor_files = (
        ("covarep", "_COVAREP.csv", 74), ("formant", "_FORMANT.csv", 5),
        ("au", "_CLNF_AUs.txt", 20), ("gaze", "_CLNF_gaze.txt", 12), ("pose", "_CLNF_pose.txt", 6),
    )
    with zipfile.ZipFile(interview) as archive:
        transcripts = [p for p in archive.namelist() if p.endswith("_TRANSCRIPT.csv")]
        if len(transcripts) != 1:
            raise ValueError("Expected one transcript per interview archive")
        prefix = Path(transcripts[0]).name.removesuffix("_TRANSCRIPT.csv")
        with archive.open(transcripts[0]) as stream:
            turns = participant_turns(stream)
        features, masks = {}, {}
        for name, suffix, width in descriptor_files:
            members = [p for p in archive.namelist() if Path(p).name == prefix + suffix]
            if len(members) > 1:
                raise ValueError(f"Duplicate descriptor file: {suffix}")
            if not members:
                features[name] = np.zeros((len(turns), width * 2), dtype=np.float32)
                masks[name] = np.zeros_like(features[name], dtype=bool)
                continue
            spec = {"header": False, "period_seconds": 0.01}
            if name in {"au", "gaze", "pose"}:
                spec = {"header": True, "timestamp": "timestamp", "confidence": "confidence",
                        "success": "success", "minimum_confidence": 0.8}
            with archive.open(members[0]) as stream:
                features[name], masks[name] = stream_aggregate(stream, spec, turns, width)
    output = {"text": encode_text(turns, encoder, device)}
    for modality, parts in (("audio", ("covarep", "formant")), ("visual", ("au", "gaze", "pose"))):
        output[modality] = np.concatenate([features[name] for name in parts], axis=1)
        output[modality + "_valid"] = np.concatenate([masks[name] for name in parts], axis=1)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interview", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    arrays = prepare(args.interview, args.encoder, args.device)
    with args.output.open("wb") as f:
        np.savez_compressed(f, **arrays)


if __name__ == "__main__":
    main()
