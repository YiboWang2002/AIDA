"""Fine-tune a DepRePlan planner or dialogue adapter on supervised JSONL records."""

import argparse
import json
import math
import os
import random
import tempfile
from pathlib import Path


BASE_MODEL = "Qwen/Qwen3.5-9B"
BASE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
TASKS = {
    "planner": {"c_method_selection", "c_session_plan", "c_turn_strategy", "c_helpfulness_rating"},
    "dialogue": {"d_reply"},
}


def read_examples(filename, role, tokenizer):
    from depreplan.prompts import encode_record

    examples = []
    with filename.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict) or record.get("task") not in TASKS[role]:
                    raise ValueError(f"expected a {role} task")
                examples.append(encode_record(record, tokenizer, max_length=8192))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid training record at line {line_number}: {exc}") from exc
    if not examples:
        raise ValueError("The training file contains no examples.")
    return examples


def load_adapter(base, initial_adapter):
    from peft import LoraConfig, TaskType, get_peft_model
    from depreplan.model import resolve_lora_targets, restore_adapter_weights

    targets = resolve_lora_targets(base)
    if initial_adapter:
        config = LoraConfig.from_pretrained(initial_adapter)
        if (config.r, config.lora_alpha, config.lora_dropout, config.bias) != (16, 32, 0.05, "none"):
            raise ValueError("The initial planner adapter must use rank 16, alpha 32 and dropout 0.05.")
        names = {name.rsplit(".", 1)[-1] for name in targets}
        if set(config.target_modules) not in (names, set(targets)) or config.modules_to_save:
            raise ValueError("The initial adapter must match the language-layer target modules.")
        config.target_modules = set(targets)
        config.inference_mode = False
    else:
        config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                            target_modules=targets, task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(base, config)
    if initial_adapter:
        restore_adapter_weights(model, initial_adapter, "default")
    for name, parameter in model.named_parameters():
        trainable = ".lora_A.default." in name or ".lora_B.default." in name
        parameter.requires_grad_(trainable)
        if trainable:
            parameter.data = parameter.data.float()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def learning_rate_factor(step, total_steps, warmup_steps):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return 0.5 * (1 + math.cos(math.pi * progress))


def target_loss(model, inputs, mask, labels):
    """Mean next-token cross entropy over this example's assistant target, including EOS."""
    import torch

    shifted = labels[0, 1:]
    positions = shifted.ne(-100).nonzero(as_tuple=True)[0]
    if positions.numel() == 0:
        raise ValueError("Training examples must contain assistant target tokens.")
    result = model(input_ids=inputs, attention_mask=mask, logits_to_keep=positions,
                   use_cache=False, return_dict=True)
    return torch.nn.functional.cross_entropy(result.logits[0].float(), shifted[positions])


def save_adapter(model, output):
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as staging:
        candidate = Path(staging) / "adapter"
        candidate.mkdir()
        config = model.peft_config["default"]
        config.base_model_name_or_path = BASE_MODEL
        config.inference_mode = True
        config.save_pretrained(candidate)
        weights = get_peft_model_state_dict(model, save_embedding_layers=False)
        save_file({key: value.detach().cpu().contiguous() for key, value in weights.items()},
                  str(candidate / "adapter_model.safetensors"), metadata={"format": "pt"})
        if output.exists():
            raise FileExistsError(output)
        candidate.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=tuple(TASKS))
    parser.add_argument("--train-file", required=True, type=Path, help="One supervised record per JSONL line.")
    parser.add_argument("--base-model", default=BASE_MODEL, help="Qwen3.5-9B Hub ID or local snapshot.")
    parser.add_argument("--initial-adapter", type=Path, help="Existing planner PEFT directory for continuation training.")
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory for the trained adapter.")
    args = parser.parse_args()
    if args.role == "dialogue" and args.initial_adapter:
        parser.error("The dialogue adapter starts from the frozen base; omit --initial-adapter.")
    if args.initial_adapter and not all((args.initial_adapter / name).is_file()
                                        for name in ("adapter_config.json", "adapter_model.safetensors")):
        parser.error("--initial-adapter must contain a PEFT configuration and adapter weights.")
    if args.output_dir.exists():
        parser.error("Choose a new output directory.")
    if not args.train_file.is_file():
        parser.error("The training JSONL file does not exist.")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("Training requires a CUDA device with BF16 support.")
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    options = {"revision": BASE_REVISION} if args.base_model == BASE_MODEL else {}
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, **options)
    tokenizer.pad_token = "<|endoftext|>"
    examples = read_examples(args.train_file, args.role, tokenizer)
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.base_model, dtype=torch.bfloat16, attn_implementation="sdpa", **options)
    base.requires_grad_(False)
    base.config.use_cache = False
    model = load_adapter(base, args.initial_adapter).to("cuda").train()
    params = [p for p in model.parameters() if p.requires_grad]
    initial_lr = 5e-6 if args.role == "planner" else 1e-5
    optimizer = torch.optim.AdamW(params, lr=initial_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    accumulation = 16
    # The reported adapters use the end of epoch one on a two-epoch schedule.
    horizon = 2 * math.ceil(len(examples) / accumulation)
    warmup = math.ceil(0.05 * horizon)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: learning_rate_factor(step, horizon, warmup))
    order = list(range(len(examples)))
    random.Random(42).shuffle(order)
    for start in range(0, len(order), accumulation):
        group = order[start:start + accumulation]
        optimizer.zero_grad(set_to_none=True)
        for index in group:
            sample = examples[index]
            batch = [torch.tensor([sample[key]], device="cuda") for key in ("input_ids", "attention_mask", "labels")]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = target_loss(model, *batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite target loss.")
            (loss / len(group)).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
    save_adapter(model, args.output_dir)
    print(f"Saved {args.role} adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
