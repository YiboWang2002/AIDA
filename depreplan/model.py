"""A shared Qwen3.5 language backbone with planner and executor adapters."""
from pathlib import Path
from threading import RLock

from .contracts import TASK_RESERVES
from .prompts import encode_prompt, special_id


BASE_MODEL = "Qwen/Qwen3.5-9B"
BASE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
LORA_LEAVES = frozenset(("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv",
                         "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
                         "gate_proj", "up_proj", "down_proj"))


def resolve_lora_targets(base):
    """Select linear layers of the language decoder, including linear attention."""
    import torch
    targets = [name for name, layer in base.named_modules()
               if "language_model.layers." in name and isinstance(layer, torch.nn.Linear)
               and name.rsplit(".", 1)[-1] in LORA_LEAVES]
    if {name.rsplit(".", 1)[-1] for name in targets} != LORA_LEAVES:
        raise ValueError("Expected a Qwen3.5 language decoder with mixed attention")
    return targets


def restore_adapter_weights(model, path, name):
    """Restore local LoRA tensors exactly, even when the backbone is bf16."""
    import torch
    from peft.utils.save_and_load import (
        get_peft_model_state_dict, load_peft_weights, set_peft_model_state_dict,
    )
    # Casting only after the copy would have already rounded the saved values.
    for key, parameter in model.named_parameters():
        if f".lora_A.{name}." in key or f".lora_B.{name}." in key:
            parameter.data = parameter.data.float()
    state = load_peft_weights(str(path), device="cpu", local_files_only=True)
    loaded = set_peft_model_state_dict(model, state, adapter_name=name)
    if loaded.unexpected_keys:
        raise ValueError(f"Unexpected weights in {name} adapter")
    actual = get_peft_model_state_dict(model, adapter_name=name)
    if set(actual) != set(state) or any(
            not torch.equal(actual[key].detach().cpu().float(), value.cpu().float())
            for key, value in state.items()):
        raise ValueError(f"Could not restore all {name} adapter tensors exactly")


class DialogueModel:
    def __init__(self, base_model, planner_adapter, executor_adapter, device="cuda:0"):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

        paths = {"planner": Path(planner_adapter), "executor": Path(executor_adapter)}
        for path in paths.values():
            for name in ("adapter_config.json", "adapter_model.safetensors"):
                if not (path / name).is_file():
                    raise FileNotFoundError(path / name)
        options = {"revision": BASE_REVISION} if str(base_model) == BASE_MODEL and not Path(base_model).is_dir() else {}
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=False, **options)
        self.tokenizer.pad_token = "<|endoftext|>"
        base = Qwen3_5ForConditionalGeneration.from_pretrained(
            base_model, dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=False, **options)
        base.requires_grad_(False)
        base.config.use_cache = True
        targets = resolve_lora_targets(base)
        for role, path in paths.items():
            config = LoraConfig.from_pretrained(str(path), local_files_only=True)
            # PEFT may serialize these as short suffixes shared by vision layers.
            configured = set(config.target_modules or ())
            if configured not in (set(LORA_LEAVES), set(targets)):
                raise ValueError(f"Unexpected language adapter target modules for {role}")
            config.target_modules = set(targets)
            if role == "planner":
                self.model = get_peft_model(base, config, adapter_name=role)
            else:
                self.model.add_adapter(role, config)
            restore_adapter_weights(self.model, path, role)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.requires_grad_(False)
        self.model.eval()
        self.lock = RLock()

    def __call__(self, record):
        import torch
        prefix = encode_prompt(record, self.tokenizer)["input_ids"]
        ids = torch.tensor([prefix], device=self.device)
        eos = special_id(self.tokenizer, "<|im_end|>")
        role = "executor" if record["task"] == "d_reply" else "planner"
        with self.lock, torch.inference_mode():
            self.model.set_adapter(role)
            self.model.requires_grad_(False)
            sequence = self.model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                max_new_tokens=TASK_RESERVES[record["task"]],
                eos_token_id=eos, pad_token_id=self.tokenizer.pad_token_id)
        generated = sequence[0, len(prefix):].tolist()
        if not generated or generated[-1] != eos:
            raise ValueError(f"Incomplete {record['task']} generation")
        return self.tokenizer.decode(generated, skip_special_tokens=True)
