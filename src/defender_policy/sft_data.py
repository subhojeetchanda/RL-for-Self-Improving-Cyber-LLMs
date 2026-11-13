"""Validation and manifest preparation for the Defender SFT warm-up corpus.

Schema (per record)
-------------------
Required fields (all must be present):
  id              – unique identifier; no overlap across splits
  scenario_id     – scenario template identifier; kept disjoint across splits
  source_type     – content origin: direct | email | log | ticket | webpage |
                    tool_result | incident_note
  attack_family   – null for benign; one of: direct_injection | indirect_injection |
                    secret_extraction | unauthorized_operation | tool_confusion
  label           – behavior target: one of the four REQUIRED_LABELS
  split           – train | validation | heldout
  user_task       – the legitimate user-facing instruction (non-empty string)
  untrusted_content – content from an untrusted source (may be empty string)
  target          – the expected safe Defender response (non-empty string)
  safety_rationale – brief explanation of the expected safe behavior (non-empty string)

Split rules (enforced)
-----------------------
- IDs must be globally disjoint across train, validation, and held-out.
- scenario_ids must be disjoint across train and held-out (to prevent template leakage).
- Training split must contain all four REQUIRED_LABELS.
- Held-out split must NOT be used for hyperparameter selection (caller's responsibility).
"""

import json
from pathlib import Path
from typing import Optional

REQUIRED_FIELDS = {
    "id",
    "scenario_id",
    "source_type",
    "attack_family",
    "label",
    "split",
    "user_task",
    "untrusted_content",
    "target",
    "safety_rationale",
}

REQUIRED_LABELS = {
    "benign_answer",
    "benign_tool_use",
    "injection_resistance",
    "unsafe_request_refusal",
}

VALID_SOURCE_TYPES = {
    "direct",
    "email",
    "log",
    "ticket",
    "webpage",
    "tool_result",
    "incident_note",
}

VALID_ATTACK_FAMILIES = {
    None,
    "direct_injection",
    "indirect_injection",
    "secret_extraction",
    "unauthorized_operation",
    "tool_confusion",
}


def load_jsonl(path: Path) -> list[dict]:
    """Load and validate a JSONL corpus file.

    Raises ``ValueError`` with a human-readable message on any schema violation.
    Does not perform cross-split checks; use ``prepare_manifest`` for those.
    """
    rows: list[dict] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from error

            # Field presence check
            row_keys = set(row)
            missing = sorted(REQUIRED_FIELDS - row_keys)
            extra = sorted(row_keys - (REQUIRED_FIELDS | {"is_novel", "quadrant"}))
            if missing or extra:
                raise ValueError(
                    f"{path}:{line_number} field mismatch — "
                    f"missing={missing}, extra={extra}"
                )

            # Label validation
            if row["label"] not in REQUIRED_LABELS:
                raise ValueError(
                    f"{path}:{line_number} unsupported label {row['label']!r}; "
                    f"must be one of {sorted(REQUIRED_LABELS)}"
                )

            # Source type validation
            if row["source_type"] not in VALID_SOURCE_TYPES:
                raise ValueError(
                    f"{path}:{line_number} unsupported source_type {row['source_type']!r}; "
                    f"must be one of {sorted(VALID_SOURCE_TYPES)}"
                )

            # Attack family validation (null is valid for benign cases)
            if row["attack_family"] not in VALID_ATTACK_FAMILIES:
                raise ValueError(
                    f"{path}:{line_number} unsupported attack_family {row['attack_family']!r}"
                )

            # Benign/attack consistency: injection_resistance and
            # unsafe_request_refusal should have an attack_family;
            # benign labels should not.
            attack_labels = {"injection_resistance", "unsafe_request_refusal"}
            if row["label"] in attack_labels and row["attack_family"] is None:
                raise ValueError(
                    f"{path}:{line_number} label {row['label']!r} requires a non-null attack_family"
                )
            if row["label"] not in attack_labels and row["attack_family"] is not None:
                raise ValueError(
                    f"{path}:{line_number} benign label {row['label']!r} "
                    f"must have attack_family=null, got {row['attack_family']!r}"
                )

            # Non-empty text fields (untrusted_content may be empty)
            text_fields = REQUIRED_FIELDS - {"untrusted_content", "attack_family"}
            for field in text_fields:
                value = row[field]
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"{path}:{line_number} field {field!r} must be a non-empty string"
                    )

            rows.append(row)
    return rows


def prepare_manifest(
    train_path: Path,
    validation_path: Path,
    heldout_path: Optional[Path] = None,
) -> dict:
    """Validate all splits and return a summary manifest.

    If ``heldout_path`` is provided, verifies ID and scenario_id disjointness
    between train and held-out (scenario leakage check).
    """
    train_rows = load_jsonl(train_path)
    validation_rows = load_jsonl(validation_path)

    train_ids = {row["id"] for row in train_rows}
    validation_ids = {row["id"] for row in validation_rows}
    train_scenario_ids = {row["scenario_id"] for row in train_rows}

    # Train/validation ID disjointness
    tv_overlap = train_ids & validation_ids
    if tv_overlap:
        raise ValueError(f"Train/validation IDs overlap: {sorted(tv_overlap)}")

    # All four behavior labels in training
    train_labels = {row["label"] for row in train_rows}
    if train_labels != REQUIRED_LABELS:
        missing = sorted(REQUIRED_LABELS - train_labels)
        raise ValueError(f"Training split is missing labels: {missing}")

    manifest: dict = {
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "train_labels": sorted(train_labels),
        "validation_labels": sorted({row["label"] for row in validation_rows}),
        "train_attack_families": sorted({r["attack_family"] for r in train_rows if r["attack_family"]}),
        "train_source_types": sorted({r["source_type"] for r in train_rows}),
        "train_validation_id_overlap": len(tv_overlap),
    }

    if heldout_path is not None:
        heldout_rows = load_jsonl(heldout_path)
        heldout_ids = {row["id"] for row in heldout_rows}
        heldout_scenario_ids = {row["scenario_id"] for row in heldout_rows}

        # ID disjointness: train vs held-out and validation vs held-out
        th_overlap = train_ids & heldout_ids
        vh_overlap = validation_ids & heldout_ids
        scenario_overlap = train_scenario_ids & heldout_scenario_ids

        if th_overlap:
            raise ValueError(f"Train/heldout IDs overlap: {sorted(th_overlap)}")
        if vh_overlap:
            raise ValueError(f"Validation/heldout IDs overlap: {sorted(vh_overlap)}")
        if scenario_overlap:
            raise ValueError(
                f"Train/heldout scenario_ids overlap (template leakage): {sorted(scenario_overlap)}"
            )

        manifest.update({
            "heldout_examples": len(heldout_rows),
            "heldout_labels": sorted({row["label"] for row in heldout_rows}),
            "heldout_attack_families": sorted({r["attack_family"] for r in heldout_rows if r["attack_family"]}),
            "train_heldout_id_overlap": len(th_overlap),
            "validation_heldout_id_overlap": len(vh_overlap),
            "train_heldout_scenario_overlap": len(scenario_overlap),
            "heldout_frozen": True,
            "heldout_note": "DO NOT tune hyperparameters using this split.",
        })

    return manifest
