#!/usr/bin/env python3
"""SFT baseline training on capstone four-quadrant data.

Trains a DistilBERT binary classifier on data/capstone/sft_train.jsonl
(Q1 common benign + Q2 common injection only — in-distribution for SFT).

Label mapping:
  - benign_answer  -> 0 (ALLOW / answer)
  - injection_resistance -> 1 (DEFEND / resist)

Saves checkpoint to checkpoints/sft_capstone/sft_final.pt
Writes metrics to reports/sft_capstone_training_metrics.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TRAIN_PATH  = PROJECT_ROOT / "data" / "capstone" / "sft_train.jsonl"
DEV_PATH    = PROJECT_ROOT / "data" / "capstone" / "sft_dev.jsonl"
TEST_PATH   = PROJECT_ROOT / "data" / "capstone" / "test.jsonl"
CKPT_DIR    = PROJECT_ROOT / "checkpoints" / "sft_capstone"
REPORTS_DIR = PROJECT_ROOT / "reports"

SAFE_LABELS   = {"benign_answer"}
UNSAFE_LABELS = {"injection_resistance"}


class CapstoneDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=128):
        self.samples = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                text = f"[TASK] {rec.get('user_task','')} [CONTENT] {rec.get('untrusted_content','')}"
                label_str = rec.get("label", "benign_answer")
                label = 1 if label_str in UNSAFE_LABELS else 0
                self.samples.append({
                    "text": text,
                    "label": label,
                    "label_str": label_str,
                    "quadrant": rec.get("quadrant", ""),
                    "is_attack": rec.get("is_attack", False),
                    "id": rec.get("id", ""),
                })
        enc = tokenizer(
            [s["text"] for s in self.samples],
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label": torch.tensor(self.samples[idx]["label"], dtype=torch.long),
            "quadrant": self.samples[idx]["quadrant"],
            "is_attack": self.samples[idx]["is_attack"],
        }


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    per_quadrant = {}
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["label"].to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            preds = logits.argmax(dim=-1).cpu().tolist()
            for pred, lbl, quad, is_atk in zip(
                preds, lbls.cpu().tolist(),
                batch["quadrant"], batch["is_attack"]
            ):
                total += 1
                if pred == lbl:
                    correct += 1
                if quad not in per_quadrant:
                    per_quadrant[quad] = {"correct": 0, "total": 0}
                per_quadrant[quad]["total"] += 1
                if pred == lbl:
                    per_quadrant[quad]["correct"] += 1
    return correct / max(total, 1), per_quadrant


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    torch.manual_seed(42)

    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[SFT] Device: {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    ).to(device)

    train_ds = CapstoneDataset(TRAIN_PATH, tokenizer, args.max_length)
    dev_ds = CapstoneDataset(DEV_PATH, tokenizer, args.max_length)
    test_ds = CapstoneDataset(TEST_PATH, tokenizer, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    print(f"[SFT] Train: {len(train_ds)} | Dev: {len(dev_ds)} | Test: {len(test_ds)}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    best_dev = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(input_ids=ids, attention_mask=mask).logits
            loss = criterion(logits, lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(lbls)
            preds = logits.argmax(dim=-1)
            train_correct += (preds == lbls).sum().item()
            train_total += len(lbls)

        avg_train_loss = train_loss / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)
        dev_acc, dev_pq = evaluate(model, dev_loader, device)
        test_acc, test_pq = evaluate(model, test_loader, device)

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 6),
            "train_accuracy": round(train_acc, 4),
            "dev_accuracy": round(dev_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "dev_per_quadrant": dev_pq,
            "test_per_quadrant": test_pq,
        })

        if epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"[SFT] Epoch {epoch:>2}/{args.epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.3f} | "
                f"Dev Acc: {dev_acc:.3f} | Test Acc: {test_acc:.3f}",
                flush=True,
            )

        if dev_acc > best_dev:
            best_dev = dev_acc
            CKPT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {k: v.detach().cpu() for k, v in model.state_dict().items()},
                CKPT_DIR / "sft_final.pt",
            )

    # Final save
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {k: v.detach().cpu() for k, v in model.state_dict().items()},
        CKPT_DIR / "sft_final.pt",
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "run_metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "algorithm": "SFT (Supervised Fine-Tuning)",
            "model": "distilbert-base-uncased",
            "n_epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "train_examples": len(train_ds),
            "dev_examples": len(dev_ds),
            "test_examples": len(test_ds),
            "data": "capstone four-quadrant (sft_train = Q1 benign + Q2 injection only)",
        },
        "best_dev_accuracy": round(best_dev, 4),
        "history": history,
    }
    out = REPORTS_DIR / "sft_capstone_training_metrics.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[SFT] Checkpoint saved → {CKPT_DIR / 'sft_final.pt'}", flush=True)
    print(f"[SFT] Metrics saved → {out}", flush=True)
    print(f"[SFT] Best dev accuracy: {best_dev:.4f}", flush=True)


if __name__ == "__main__":
    main()
