"""Loading helpers for the capstone corpora.

Every consumer (SFT trainer, MAPPO trainer, evaluator, API server) reads the
JSONL files written by ``scripts/build_capstone_data.py`` through these
functions, so the label encoding can never drift between training and
evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "capstone"

#: Action space, shared by training, evaluation and serving.
ALLOW: int = 0
DEFEND: int = 1

QUADRANTS: tuple[str, ...] = (
    "Q1_common_benign",
    "Q2_common_injection",
    "Q3_unique_benign",
    "Q4_unique_injection",
)

QUADRANT_META: dict[str, dict[str, str]] = {
    "Q1_common_benign": {
        "label": "Common benign",
        "description": "Ordinary request. Both guards must ALLOW.",
        "expected_action": "ALLOW",
        "sft_expected": "ALLOW",
        "mappo_expected": "ALLOW",
    },
    "Q2_common_injection": {
        "label": "Common injection",
        "description": "Overt in-distribution injection. Both guards must DEFEND.",
        "expected_action": "DEFEND",
        "sft_expected": "DEFEND",
        "mappo_expected": "DEFEND",
    },
    "Q3_unique_benign": {
        "label": "Unique benign (OOD)",
        "description": (
            "Benign request that carries attack-looking vocabulary. The SFT "
            "baseline false-positives; MAPPO must ALLOW."
        ),
        "expected_action": "ALLOW",
        "sft_expected": "DEFEND",
        "mappo_expected": "ALLOW",
    },
    "Q4_unique_injection": {
        "label": "Unique injection (OOD)",
        "description": (
            "Extraction attempt without the SFT attack vocabulary. The SFT "
            "baseline false-negatives; MAPPO must DEFEND."
        ),
        "expected_action": "DEFEND",
        "sft_expected": "ALLOW",
        "mappo_expected": "DEFEND",
    },
}


def label_to_action(is_attack: bool) -> int:
    """Target action for a case: attack -> DEFEND, benign -> ALLOW."""
    return DEFEND if is_attack else ALLOW


@dataclass(frozen=True)
class Record:
    id: str
    quadrant: str
    split: str
    user_task: str
    untrusted_content: str
    is_attack: bool
    attack_family: str | None
    policy_input: str
    targets_secret: bool
    contains_attack_lexicon: bool

    @classmethod
    def from_dict(cls, raw: dict) -> "Record":
        return cls(
            id=raw["id"],
            quadrant=raw["quadrant"],
            split=raw.get("split", ""),
            user_task=raw.get("user_task", ""),
            untrusted_content=raw.get("untrusted_content", ""),
            is_attack=bool(raw["is_attack"]),
            attack_family=raw.get("attack_family"),
            policy_input=raw["policy_input"],
            targets_secret=bool(raw.get("targets_secret", False)),
            contains_attack_lexicon=bool(raw.get("contains_attack_lexicon", False)),
        )

    @property
    def target_action(self) -> int:
        return label_to_action(self.is_attack)


def load_jsonl(path: Path | str) -> list[Record]:
    p = Path(path)
    records: list[Record] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(Record.from_dict(json.loads(line)))
    return records


def load_corpus(name: str, data_dir: Path | str = DATA_DIR) -> list[Record]:
    """Load one of the standard splits: ``sft_train``, ``sft_dev``,
    ``mappo_benign``, ``mappo_attack``, ``test``."""
    return load_jsonl(Path(data_dir) / f"{name}.jsonl")


def by_quadrant(records: Iterable[Record], quadrant: str) -> list[Record]:
    return [r for r in records if r.quadrant == quadrant]


def action_name(action: int) -> str:
    return "DEFEND" if int(action) == DEFEND else "ALLOW"


def counts(records: Sequence[Record]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in records:
        out[r.quadrant] = out.get(r.quadrant, 0) + 1
    return out
