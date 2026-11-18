#!/usr/bin/env python3
"""Materialise the capstone corpora under ``data/capstone/``.

Outputs
-------
    data/capstone/sft_train.jsonl      Q1 benign + Q2 injection (in-distribution only)
    data/capstone/sft_dev.jsonl        in-distribution dev split (model selection)
    data/capstone/mappo_benign.jsonl   Q1 + Q3 benign pool for co-evolution
    data/capstone/mappo_attack.jsonl   Q2 + Q4 injection pool for co-evolution
    data/capstone/test.jsonl           the frozen 100-case four-quadrant benchmark
    data/capstone/corpus_stats.json    facts + digests that justify the contrast

The files are derived from :mod:`src.capstone.corpus`; this script never invents
data of its own.  Re-running it is idempotent for a fixed seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.capstone.corpus import build_splits, corpus_statistics  # noqa: E402
from src.capstone.task import (  # noqa: E402
    DEFAULT_SECRET,
    build_policy_input,
    build_system_prompt,
    scan_for_secret,
    targets_secret,
)

DEFAULT_OUT = PROJECT_ROOT / "data" / "capstone"


def _record(case, split: str) -> dict:
    return {
        "id": case.id,
        "quadrant": case.quadrant,
        "split": split,
        "user_task": case.user_task,
        "untrusted_content": case.content,
        "label": "injection_resistance" if case.is_attack else "benign_answer",
        "is_attack": case.is_attack,
        "attack_family": case.family,
        "policy_input": build_policy_input(case.user_task, case.content),
        "targets_secret": targets_secret(case.content) if case.is_attack else False,
        "contains_attack_lexicon": _lexicon_flag(case.content),
    }


def _lexicon_flag(text: str) -> bool:
    from src.capstone.corpus import has_lexicon

    return has_lexicon(text)


def _write(path: Path, records: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    payload = path.read_bytes()
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--secret", default=DEFAULT_SECRET,
                        help="Token embedded in the guarded agent's system prompt.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    splits = build_splits(seed=args.seed)

    files = {
        "sft_train.jsonl": [_record(c, "sft_train") for c in splits.sft_train],
        "sft_dev.jsonl": [_record(c, "dev") for c in splits.dev],
        "mappo_benign.jsonl": [_record(c, "mappo_train") for c in splits.mappo_benign_train],
        "mappo_attack.jsonl": [_record(c, "mappo_train") for c in splits.mappo_attack_train],
        "test.jsonl": [_record(c, "test") for c in splits.test],
    }

    stats = corpus_statistics(splits)
    stats.update({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "secret": args.secret,
        "system_prompt": build_system_prompt(args.secret),
        "policy_input_template": "[TASK] {user_task} [CONTENT] {untrusted_content}",
        "secret_in_guard_input": scan_for_secret(build_system_prompt(args.secret), args.secret),
        "file_digests": {},
    })

    print("=" * 74)
    print("CAPSTONE CORPORA")
    print("=" * 74)
    for name, records in files.items():
        attacks = sum(1 for r in records if r["is_attack"])
        print(f"  {name:<22} {len(records):>4} records   benign={len(records)-attacks:>3} "
              f"attack={attacks:>3}")
        stats["file_digests"][name] = _digest_preview(records)

    print("-" * 74)
    for key in ("sft_benign_with_attack_lexicon", "sft_attack_with_attack_lexicon",
                "q3_benign_with_attack_lexicon", "q4_injection_with_attack_lexicon",
                "q2_targets_secret", "q4_targets_secret"):
        print(f"  {key:<36} {stats[key]}")
    print(f"  {'test_digest':<36} {stats['test_digest']}")
    print("=" * 74)

    if args.dry_run:
        print("dry run: nothing written")
        return 0

    for name, records in files.items():
        stats["file_digests"][name] = _write(out_dir / name, records)
    stats_path = out_dir / "corpus_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"wrote {len(files) + 1} files to {out_dir}")
    return 0


def _digest_preview(records: list[dict]) -> str:
    blob = "\n".join(sorted(json.dumps(r, sort_keys=True) for r in records))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


if __name__ == "__main__":
    raise SystemExit(main())
