"""scripts/generate_review2_report.py — Member 4: Review 2 report aggregator.

Runs N fixture coevolution episodes using CoevolutionEnv (with the dummy
attacker and FixtureModelAdapter), logs all step/episode metrics through
CoevolutionWandbLogger, and writes:

    reports/review2_coevolution_metrics.json   — ASR vs. step, per-family stats
    reports/review2_adversarial_traces.json    — successful vs. defended samples

Usage
-----
    python3 scripts/generate_review2_report.py [OPTIONS]

Options
-------
    --n-episodes N       Episodes to run (default: 25 — one per review2 case)
    --seed SEED          RNG seed (default: 42)
    --output-dir DIR     Output directory (default: reports/)
    --wandb-mode MODE    W&B mode: online|offline|disabled (default: disabled)
    --dry-run            Run episodes but write no files
    --verbose            Print per-episode summary to stdout
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from coevolution_env import CoevolutionEnv                                # noqa: E402
from evaluation_metrics.wandb_logger import CoevolutionWandbLogger        # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Review 2 coevolution reports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--n-episodes", type=int, default=25,
        help="Number of fixture coevolution episodes to run (default: 25).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible case sampling (default: 42).",
    )
    parser.add_argument(
        "--output-dir", default=str(PROJECT_ROOT / "reports"),
        help="Directory for output JSON reports (default: reports/).",
    )
    parser.add_argument(
        "--wandb-mode", default="disabled",
        choices=["online", "offline", "disabled"],
        help="Weights & Biases mode (default: disabled — writes JSON only).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run episodes but do not write any output files.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-episode summary to stdout.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    run_id = f"review2_coevo_seed{args.seed}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"

    print(f"[generate_review2_report] run_id={run_id}")
    print(f"  Episodes : {args.n_episodes}")
    print(f"  Seed     : {args.seed}")
    print(f"  Output   : {output_dir}")
    print(f"  W&B mode : {args.wandb_mode}")
    print(f"  Dry-run  : {args.dry_run}")
    print()

    # ── Initialise env and logger ─────────────────────────────────────
    env = CoevolutionEnv(seed=args.seed)

    wandb_logger = CoevolutionWandbLogger(
        output_dir=output_dir,
        run_name=run_id,
        mode=args.wandb_mode,
    )
    wandb_logger.init(
        tags=["review2", "coevolution", "fixture"],
        extra_config={
            "n_episodes": args.n_episodes,
            "seed": args.seed,
            "defender": "FixtureModelAdapter",
            "attacker": "DummyAttacker",
        },
    )

    # ── Run episodes ──────────────────────────────────────────────────
    global_step = 0
    all_episode_results = []

    for ep_idx in range(args.n_episodes):
        ep_id = f"ep_{ep_idx:04d}"
        episode_result = env.run_episode(episode_id=ep_id)
        all_episode_results.append(episode_result)

        # Log each step then the episode summary
        global_step = wandb_logger.log_steps_from_episode(
            episode_result,
            base_step=global_step,
        )

        if args.verbose:
            _print_episode_summary(episode_result, ep_idx)

    env.close()

    # ── Aggregate console summary ─────────────────────────────────────
    total_eps = len(all_episode_results)
    attack_eps = [r for r in all_episode_results if r.case_kind == "attack"]
    benign_eps = [r for r in all_episode_results if r.case_kind == "benign"]

    mean_asr = (
        sum(r.attack_success_rate for r in attack_eps) / len(attack_eps)
        if attack_eps else 0.0
    )
    mean_def_reward = (
        sum(r.total_defender_reward for r in all_episode_results) / total_eps
        if total_eps else 0.0
    )
    mean_gate_blocks = (
        sum(r.gate_block_count for r in all_episode_results) / total_eps
        if total_eps else 0.0
    )

    print(f"  Episodes run        : {total_eps}")
    print(f"  Attack episodes     : {len(attack_eps)}")
    print(f"  Benign episodes     : {len(benign_eps)}")
    print(f"  Mean ASR (attack)   : {mean_asr:.4f}")
    print(f"  Mean defender reward: {mean_def_reward:.4f}")
    print(f"  Mean gate blocks/ep : {mean_gate_blocks:.2f}")
    print()

    if args.dry_run:
        print("[dry-run] Episodes complete — no files written.")
        wandb_logger._tracker.finish()
        return

    # ── Write reports ─────────────────────────────────────────────────
    written = wandb_logger.flush_reports()
    wandb_logger.finish()

    for name, path in written.items():
        print(f"  [{name}] → {path}")

    print()
    print("[DONE] Review 2 report generation complete.")
    print("  All results are fixture baseline. Do not present as trained-LLM results.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_episode_summary(episode_result: Any, ep_idx: int) -> None:
    """Pretty-print a one-line episode summary."""
    family = episode_result.attack_family or "benign"
    print(
        f"  ep={ep_idx:03d} | {episode_result.case_id:<20s} | "
        f"kind={episode_result.case_kind:<7s} | family={family:<22s} | "
        f"steps={episode_result.step_count} | "
        f"ASR={episode_result.attack_success_rate:.2f} | "
        f"R_def={episode_result.total_defender_reward:+.3f} | "
        f"gates={episode_result.gate_block_count}"
    )


if __name__ == "__main__":
    from typing import Any   # needed in _print_episode_summary at runtime
    main()
