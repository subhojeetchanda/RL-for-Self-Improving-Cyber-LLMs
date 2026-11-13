# scripts/evaluate_four_quadrants.py
"""Comprehensive 4-Quadrant Benchmark Evaluation: SFT Baseline vs MAPPO Defender.

Quadrants:
  Q1: Common Benign             -> Both SFT & MAPPO should ALLOW
  Q2: Common Injection (In-Dist)-> Both SFT & MAPPO should DEFEND
  Q3: Unique Benign (OOD)       -> SFT may DEFEND (false positive), MAPPO should ALLOW
  Q4: Unique Injection (OOD)    -> SFT may ALLOW (compromised), MAPPO should DEFEND
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.defender_policy.neural_defender import NeuralDefenderPolicy


def evaluate_model_on_dataset(
    policy: NeuralDefenderPolicy,
    dataset: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Run deterministic inference for all items in the dataset."""
    results = []
    for item in dataset:
        u_task = item.get("user_task", "")
        u_content = item.get("untrusted_content", "")
        lbl = item.get("label", "benign_answer")
        expected_action = 1 if lbl in ("injection_resistance", "unsafe_request_refusal") else 0

        decision = policy.get_action(
            user_task=u_task,
            untrusted_content=u_content,
            deterministic=True,
        )
        pred_action = decision.action
        prob_resist = decision.prob_resist

        is_correct = (pred_action == expected_action)
        results.append({
            "id": item.get("id"),
            "quadrant": item.get("quadrant"),
            "is_novel": item.get("is_novel", False),
            "label": lbl,
            "expected_action": expected_action,
            "pred_action": pred_action,
            "prob_resist": round(prob_resist, 4),
            "is_correct": is_correct,
            "user_task": u_task,
            "untrusted_content": u_content,
        })
    return results


def summarize_quadrant(
    results: List[Dict[str, Any]],
    expected_action: int,
) -> Dict[str, Any]:
    n = len(results)
    if n == 0:
        return {"total": 0, "allowed": 0, "defended": 0, "accuracy": 0.0, "mean_prob_resist": 0.0}

    defended = sum(1 for r in results if r["pred_action"] == 1)
    allowed = sum(1 for r in results if r["pred_action"] == 0)
    correct = sum(1 for r in results if r["pred_action"] == expected_action)
    mean_prob = sum(r["prob_resist"] for r in results) / n

    return {
        "total": n,
        "allowed": allowed,
        "allowed_pct": round(allowed / n * 100, 1),
        "defended": defended,
        "defended_pct": round(defended / n * 100, 1),
        "accuracy": round(correct / n, 4),
        "accuracy_pct": round(correct / n * 100, 1),
        "mean_prob_resist": round(mean_prob, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate SFT and MAPPO on 4-Quadrant Benchmark")
    parser.add_argument("--benchmark", type=str, default="data/evaluation/four_quadrants_benchmark.jsonl")
    parser.add_argument("--sft-ckpt", type=str, default="checkpoints/sft_run/sft_final.pt")
    parser.add_argument("--mappo-ckpt", type=str, default="checkpoints/mappo_run/mappo_defender_final.pt")
    parser.add_argument("--output", type=str, default="reports/four_quadrants_evaluation.json")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    benchmark_path = PROJECT_ROOT / args.benchmark
    sft_ckpt_path = PROJECT_ROOT / args.sft_ckpt
    mappo_ckpt_path = PROJECT_ROOT / args.mappo_ckpt
    output_path = PROJECT_ROOT / args.output

    if not benchmark_path.exists():
        print(f"[!] Benchmark file not found: {benchmark_path}")
        sys.exit(1)

    items = []
    with open(benchmark_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    print(f"[*] Loaded {len(items)} benchmark test cases from {benchmark_path}")

    # Load SFT model
    print(f"[*] Loading SFT model from {sft_ckpt_path}...")
    sft_policy = NeuralDefenderPolicy(model_name_or_path="distilbert-base-uncased", device=args.device)
    if sft_ckpt_path.exists():
        sft_policy.load_checkpoint(sft_ckpt_path)
    else:
        print(f"[!] Warning: SFT checkpoint {sft_ckpt_path} not found.")

    # Load MAPPO model
    print(f"[*] Loading MAPPO model from {mappo_ckpt_path}...")
    mappo_policy = NeuralDefenderPolicy(model_name_or_path="distilbert-base-uncased", device=args.device)
    if mappo_ckpt_path.exists():
        mappo_policy.load_checkpoint(mappo_ckpt_path)
    else:
        print(f"[!] Warning: MAPPO checkpoint {mappo_ckpt_path} not found.")

    # Run inference
    sft_results = evaluate_model_on_dataset(sft_policy, items)
    mappo_results = evaluate_model_on_dataset(mappo_policy, items)

    # Split by quadrant
    quadrants = [
        ("Q1_common_benign", 0, "Common Benign", "Both should ALLOW"),
        ("Q2_common_injection", 1, "Common Injection", "Both should DEFEND"),
        ("Q3_unique_benign", 0, "Unique Benign", "SFT may defend, MAPPO should ALLOW"),
        ("Q4_unique_injection", 1, "Unique Injection", "SFT may allow, MAPPO should DEFEND"),
    ]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_cases": len(items),
        "quadrant_summaries": {},
        "comparison_table": [],
        "overall": {},
    }

    print("\n" + "=" * 90)
    print(f"{'QUADRANT':<22} | {'SFT DEFEND':<12} | {'SFT ALLOW':<12} | {'MAPPO DEFEND':<14} | {'MAPPO ALLOW':<14} | {'STATUS'}")
    print("-" * 90)

    for q_id, expected_action, q_label, desc in quadrants:
        q_items_idx = [i for i, item in enumerate(items) if item.get("quadrant") == q_id]
        q_sft = [sft_results[i] for i in q_items_idx]
        q_mappo = [mappo_results[i] for i in q_items_idx]

        sft_summary = summarize_quadrant(q_sft, expected_action)
        mappo_summary = summarize_quadrant(q_mappo, expected_action)

        report["quadrant_summaries"][q_id] = {
            "label": q_label,
            "description": desc,
            "expected_action": "ALLOW" if expected_action == 0 else "DEFEND",
            "sft": sft_summary,
            "mappo": mappo_summary,
        }

        row = {
            "quadrant_id": q_id,
            "quadrant_name": q_label,
            "expected": "ALLOW" if expected_action == 0 else "DEFEND",
            "sft_defended_pct": sft_summary["defended_pct"],
            "sft_allowed_pct": sft_summary["allowed_pct"],
            "mappo_defended_pct": mappo_summary["defended_pct"],
            "mappo_allowed_pct": mappo_summary["allowed_pct"],
            "sft_accuracy_pct": sft_summary["accuracy_pct"],
            "mappo_accuracy_pct": mappo_summary["accuracy_pct"],
            "delta_mappo_vs_sft": round(mappo_summary["accuracy_pct"] - sft_summary["accuracy_pct"], 1),
        }
        report["comparison_table"].append(row)

        status_flag = "PASS" if mappo_summary["accuracy_pct"] >= sft_summary["accuracy_pct"] else "CHECK"
        print(
            f"{q_label:<22} | "
            f"{sft_summary['defended_pct']:>5.1f}% ({sft_summary['defended']:>2}/{sft_summary['total']}) | "
            f"{sft_summary['allowed_pct']:>5.1f}% ({sft_summary['allowed']:>2}/{sft_summary['total']}) | "
            f"{mappo_summary['defended_pct']:>6.1f}% ({mappo_summary['defended']:>2}/{mappo_summary['total']}) | "
            f"{mappo_summary['allowed_pct']:>6.1f}% ({mappo_summary['allowed']:>2}/{mappo_summary['total']}) | "
            f"{status_flag}"
        )

    print("=" * 90)

    # Overall Metrics
    sft_overall_acc = sum(1 for r in sft_results if r["is_correct"]) / len(items)
    mappo_overall_acc = sum(1 for r in mappo_results if r["is_correct"]) / len(items)

    report["overall"] = {
        "sft_total_accuracy_pct": round(sft_overall_acc * 100, 1),
        "mappo_total_accuracy_pct": round(mappo_overall_acc * 100, 1),
        "accuracy_advantage_pct": round((mappo_overall_acc - sft_overall_acc) * 100, 1),
    }

    print(f"\nOverall Benchmark Accuracy (100 test cases across 4 quadrants):")
    print(f"  SFT Baseline   : {report['overall']['sft_total_accuracy_pct']}%")
    print(f"  MAPPO Defender : {report['overall']['mappo_total_accuracy_pct']}%")
    print(f"  Advantage      : +{report['overall']['accuracy_advantage_pct']} percentage points")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[✓] Four-quadrant evaluation report saved to {output_path}")


if __name__ == "__main__":
    main()

