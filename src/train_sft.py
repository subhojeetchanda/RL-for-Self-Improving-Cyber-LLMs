"""Train a response-generating causal language model with completion-only SFT.

The structured records under ``data/sft`` remain the source of truth. Prompt
tokens are masked from the loss; only the target response contributes to SFT.
The saved directory is a PEFT adapter accepted by ``SFTModelAdapter``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import peft
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.defender_policy.training_context import (
    DEFAULT_CANARY_LEAK_PENALTY,
    build_protected_prompt,
    protected_validation_score,
    synthetic_protected_value,
)
from src.defender_policy.model_adapter import parse_decision_response
from src.defender_policy.generation_config import DO_SAMPLE, TEMPERATURE, TOP_P


LABEL_ACTION = {
    "benign_answer": "ALLOW",
    "benign_tool_use": "ALLOW",
    "injection_resistance": "DEFEND",
    "unsafe_request_refusal": "DEFEND",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def case_fingerprints(rows: list[dict]) -> dict[str, list[str]]:
    def digest(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    return {
        "ids": sorted(digest(row["id"]) for row in rows if row.get("id")),
        "scenarios": sorted(digest(row["scenario_id"]) for row in rows if row.get("scenario_id")),
        "inputs": sorted(digest(json.dumps([
            row.get("user_task", "").strip(), row.get("untrusted_content", "").strip()
        ], ensure_ascii=False, separators=(",", ":"))) for row in rows),
    }


def load_records(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    for row in rows:
        if not isinstance(row.get("id"), str) or not row["id"].strip():
            raise ValueError(f"Every record must have a non-empty string ID in {path}")
        if row.get("scenario_id") is not None and not isinstance(row["scenario_id"], str):
            raise ValueError(f"scenario_id must be a string in {path} ({row['id']})")
        if row.get("attack_family") is not None and not isinstance(row["attack_family"], str):
            raise ValueError(f"attack_family must be a string or null in {path} ({row['id']})")
        if row.get("task_family") is not None and not isinstance(row["task_family"], str):
            raise ValueError(f"task_family must be a string or null in {path} ({row['id']})")
    for key in ("id", "scenario_id"):
        values = [row.get(key) for row in rows if row.get(key)]
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate {key} within split: {path}")
    inputs = [(row.get("user_task", "").strip(), row.get("untrusted_content", "").strip())
              for row in rows]
    if len(inputs) != len(set(inputs)):
        raise ValueError(f"Duplicate task/content pair within split: {path}")
    for row in rows:
        for field in ("user_task", "untrusted_content"):
            if not isinstance(row.get(field, ""), str):
                raise ValueError(f"Field {field!r} must be a string in {path} ({row.get('id', '<unknown>')})")
        if row.get("label") not in LABEL_ACTION:
            raise ValueError(f"Unsupported label {row.get('label')!r} in {path}")
        if not isinstance(row.get("target"), str) or not row["target"].strip():
            raise ValueError(f"Missing target response for {row.get('id', '<unknown>')} in {path}")
    return rows


def validate_disjoint_splits(train_rows: list[dict], validation_rows: list[dict]) -> None:
    for key in ("id", "scenario_id"):
        train_values = {row.get(key) for row in train_rows if row.get(key)}
        overlap = train_values.intersection(row.get(key) for row in validation_rows if row.get(key))
        if overlap:
            raise ValueError(f"{key} overlap between train and validation: {sorted(overlap)[0]}")
    train_inputs = {
        (row.get("user_task", "").strip(), row.get("untrusted_content", "").strip())
        for row in train_rows
    }
    for row in validation_rows:
        fingerprint = (row.get("user_task", "").strip(), row.get("untrusted_content", "").strip())
        if fingerprint in train_inputs:
            raise ValueError(f"Exact task/content duplicate between splits: {row.get('id', '<unknown>')}")


class CompletionDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int,
                 seed: int, namespace: str):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.seed = seed
        self.namespace = namespace
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        protected_index = index + 1
        if self.namespace == "training":
            protected_index += self.epoch * len(self.rows)
        protected_value = synthetic_protected_value(
            self.seed, protected_index, self.namespace
        )
        prompt = build_protected_prompt(
            row.get("user_task", ""), row.get("untrusted_content", ""), protected_value
        )
        completion = (
            f" DECISION: {LABEL_ACTION[row['label']]}\n{row['target']}"
            + self.tokenizer.eos_token
        )
        prompt_ids = self.tokenizer(
            prompt, add_special_tokens=True, truncation=False
        )["input_ids"]
        completion_ids = self.tokenizer(
            completion, add_special_tokens=False, truncation=False
        )["input_ids"]
        if len(prompt_ids) + len(completion_ids) > self.max_length:
            raise ValueError(
                f"Record {row.get('id')} requires {len(prompt_ids) + len(completion_ids)} "
                f"tokens, exceeding max_length={self.max_length}; shorten the record "
                "or increase --max-length so its task and full response are preserved"
            )
        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids
        if not completion_ids:
            raise ValueError(f"Completion was truncated to zero tokens for {row.get('id')}")
        return {"input_ids": input_ids, "labels": labels}


def collate(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(len(item["input_ids"]) for item in batch)
    ids, labels, masks = [], [], []
    for item in batch:
        pad = width - len(item["input_ids"])
        ids.append(item["input_ids"] + [pad_id] * pad)
        labels.append(item["labels"] + [-100] * pad)
        masks.append([1] * len(item["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
    }


@torch.no_grad()
def validation_loss(model, loader, device) -> float:
    model.eval()
    losses = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        losses.append(float(model(**batch).loss.item()))
    return sum(losses) / max(len(losses), 1)


@torch.no_grad()
def validation_decision_metrics(model, rows: list[dict], tokenizer, device: str,
                                max_prompt_tokens: int, max_new_tokens: int,
                                samples_per_case: int, seed: int,
                                leak_penalty: float) -> dict[str, float]:
    model.eval()
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    correct = 0
    exposures = 0
    total = len(rows) * samples_per_case
    try:
        for index, row in enumerate(rows):
            expected = LABEL_ACTION[row["label"]]
            expected_id = 0 if expected == "ALLOW" else 1
            for sample in range(samples_per_case):
                sample_index = index * samples_per_case + sample + 1
                protected_value = synthetic_protected_value(seed, sample_index, "validation")
                prompt = build_protected_prompt(
                    row.get("user_task", ""), row.get("untrusted_content", ""), protected_value
                )
                encoded = tokenizer(
                    prompt, return_tensors="pt", truncation=False,
                ).to(device)
                if encoded["input_ids"].shape[1] > max_prompt_tokens:
                    raise ValueError(
                        f"Validation case {row.get('id')} exceeds the generation input "
                        f"budget of {max_prompt_tokens} tokens"
                    )
                prompt_length = encoded["input_ids"].shape[1]
                sample_seed = seed + sample_index - 1
                torch.manual_seed(sample_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(sample_seed)
                generated = model.generate(
                    **encoded, max_new_tokens=max_new_tokens, do_sample=DO_SAMPLE,
                    temperature=TEMPERATURE, top_p=TOP_P,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                completion = tokenizer.decode(
                    generated[0, prompt_length:], skip_special_tokens=True
                ).strip()
                decision, body = parse_decision_response(completion)
                predicted_id = None if decision is None else int(decision == "DEFEND")
                has_response_body = bool(body)
                correct += int(predicted_id == expected_id and has_response_body)
                exposures += int(protected_value in completion)
    finally:
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
    accuracy = correct / max(total, 1)
    exposure_rate = exposures / max(total, 1)
    return {
        "decision_accuracy": accuracy,
        "synthetic_canary_exposure_rate": exposure_rate,
        "selection_score": protected_validation_score(accuracy, exposure_rate, leak_penalty),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--model-revision", default=None,
                        help="Optional immutable model revision; the resolved commit is recorded in the manifest.")
    parser.add_argument("--train-file", type=Path, default=ROOT / "data/sft/train.jsonl")
    parser.add_argument("--validation-file", type=Path, default=ROOT / "data/sft/validation.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints/sft_llm")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--validation-samples", type=int, default=5)
    parser.add_argument("--canary-leak-penalty", type=float,
                        default=DEFAULT_CANARY_LEAK_PENALTY)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.gradient_accumulation_steps) < 1:
        parser.error("epochs, batch-size, and gradient-accumulation-steps must be positive")
    if (args.learning_rate <= 0 or args.max_length < 32
            or not 1 <= args.max_new_tokens < 2048 or args.max_length <= args.max_new_tokens
            or args.validation_samples < 1):
        parser.error("learning-rate, lengths, and validation-samples must be positive; max-length must exceed max-new-tokens and max-new-tokens must be below 2048")
    if args.canary_leak_penalty <= 0:
        parser.error("canary-leak-penalty must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a new path: {args.output_dir}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")

    train_rows = load_records(args.train_file)
    validation_rows = load_records(args.validation_file)
    validate_disjoint_splits(train_rows, validation_rows)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, revision=args.model_revision)
    if tokenizer.eos_token is None:
        raise ValueError("The base tokenizer must define an EOS token for response training")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, revision=args.model_revision, torch_dtype=torch.float32
    )
    resolved_model_revision = model.config._commit_hash or args.model_revision
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )).to(device)

    train_ds = CompletionDataset(
        train_rows, tokenizer, args.max_length, args.seed, "training"
    )
    val_ds = CompletionDataset(
        validation_rows, tokenizer, args.max_length, args.seed, "validation"
    )
    collator = lambda batch: collate(batch, tokenizer.pad_token_id)
    train_actions = [LABEL_ACTION[row["label"]] for row in train_rows]
    action_counts = {action: train_actions.count(action) for action in ("ALLOW", "DEFEND")}
    if any(action_counts[action] == 0 for action in ("ALLOW", "DEFEND")):
        raise ValueError("SFT training data must contain both ALLOW and DEFEND examples")
    sample_weights = [1.0 / action_counts[action] for action in train_actions]
    train_sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(train_ds), replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                              collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    best_val_loss = float("inf")
    best_val_score = float("-inf")
    best_val_metrics = None
    best_epoch = None
    history = []
    for epoch in range(1, args.epochs + 1):
        train_ds.set_epoch(epoch - 1)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running, updates = 0.0, 0
        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            running += float(loss.detach().item())
            window_start = ((step - 1) // args.gradient_accumulation_steps) * args.gradient_accumulation_steps
            window_size = min(args.gradient_accumulation_steps, len(train_loader) - window_start)
            (loss / window_size).backward()
            if step % args.gradient_accumulation_steps == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
        val_loss = validation_loss(model, val_loader, device)
        if not math.isfinite(val_loss):
            raise RuntimeError(f"Validation loss is not finite at epoch {epoch}")
        val_metrics = validation_decision_metrics(
            model, validation_rows, tokenizer, device,
            max(1, 2048 - args.max_new_tokens), args.max_new_tokens,
            args.validation_samples, args.seed, args.canary_leak_penalty,
        )
        record = {"epoch": epoch, "train_loss_mean_batch": running / len(train_loader),
                  "validation_loss": val_loss,
                  "validation_decision_accuracy": val_metrics["decision_accuracy"],
                  "validation_synthetic_canary_exposure_rate": val_metrics["synthetic_canary_exposure_rate"],
                  "validation_selection_score": val_metrics["selection_score"],
                  "validation_samples_per_case": args.validation_samples,
                  "optimizer_updates": updates}
        history.append(record)
        print(json.dumps(record), flush=True)
        if (val_metrics["selection_score"] > best_val_score
                or (val_metrics["selection_score"] == best_val_score and val_loss < best_val_loss)):
            best_val_score = val_metrics["selection_score"]
            best_val_metrics = val_metrics
            best_val_loss = val_loss
            best_epoch = epoch
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "completion-only causal language model SFT with LoRA",
        "base_model": args.model_name,
        "base_model_revision": resolved_model_revision,
        "seed": args.seed,
        "device": device,
        "platform": platform.platform(),
        "accelerator_name": torch.cuda.get_device_name(0) if device == "cuda" else (
            "Apple MPS" if device == "mps" else platform.processor() or "CPU"
        ),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "peft_version": peft.__version__,
        "epochs_requested": args.epochs,
        "checkpoint_selection_metric": "2 * stochastic validation decision accuracy - 1 - canary leak penalty * exact synthetic-canary exposure rate; validation loss breaks ties",
        "selected_epoch": best_epoch,
        "selected_validation_decision_accuracy": best_val_metrics["decision_accuracy"],
        "selected_validation_synthetic_canary_exposure_rate": best_val_metrics["synthetic_canary_exposure_rate"],
        "selected_validation_score": best_val_score,
        "selected_validation_loss": best_val_loss,
        "validation_samples_per_case": args.validation_samples,
        "validation_seed": args.seed,
        "canary_leak_penalty": args.canary_leak_penalty,
        "validation_protected_value_namespace": "validation",
        "training_protected_value_namespace": "training",
        "training_protected_value_schedule": "one deterministic value per record per epoch",
        "validation_protected_value_schedule": "one deterministic value per case and sample",
        "adapter_weights_sha256": sha256_file(next(
            path for path in (args.output_dir / "adapter_model.safetensors", args.output_dir / "adapter_model.bin")
            if path.is_file()
        )),
        "train_examples": len(train_rows),
        "train_action_counts": action_counts,
        "train_sampling": "class-balanced ALLOW/DEFEND sampling with replacement (50/50 in expectation)",
        "synthetic_protected_value_training": "one deterministic fresh value per record per epoch; values are not written to artifacts",
        "validation_examples": len(validation_rows),
        "case_fingerprints": {
            "train": case_fingerprints(train_rows),
            "validation": case_fingerprints(validation_rows),
        },
        "train_attack_families": sorted({row["attack_family"] for row in train_rows if row.get("attack_family")}),
        "validation_attack_families": sorted({row["attack_family"] for row in validation_rows if row.get("attack_family")}),
        "train_task_families": sorted({row["task_family"] for row in train_rows if row.get("task_family")}),
        "validation_task_families": sorted({row["task_family"] for row in validation_rows if row.get("task_family")}),
        "train_file_sha256": sha256_file(args.train_file),
        "validation_file_sha256": sha256_file(args.validation_file),
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "history": history,
    }
    (args.output_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved best validation checkpoint and manifest to {args.output_dir}")


if __name__ == "__main__":
    main()
