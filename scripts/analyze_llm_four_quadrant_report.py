"""Compute the full security/generalization metric suite from four-quadrant reports.

Consumes one or more JSON reports produced by `evaluate_llm_four_quadrants.py`
and emits a machine-readable aggregate plus a Markdown summary. It does not
re-run models and never sees prompt text, responses, or canary values.

Usage:
    python3 scripts/analyze_llm_four_quadrant_report.py \
        reports/llm_eval_seed42.json [reports/llm_eval_seed43.json ...] \
        --output reports/full_metrics.json --markdown reports/full_metrics.md
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

Q_BENIGN = ("Q1_common_benign", "Q3_unique_benign")
Q_ATTACK = ("Q2_common_injection", "Q4_unique_injection")
Q_ID = ("Q1_common_benign", "Q2_common_injection")
Q_OOD = ("Q3_unique_benign", "Q4_unique_injection")
Q_ALL = Q_ID + Q_OOD


def _mean_of_quadrants(summary, arm, quadrants, key):
    values = [summary[q][arm].get(key) for q in quadrants]
    values = [v for v in values if isinstance(v, (int, float))]
    return sum(values) / len(values) if values else None


def _pooled_rate(summary, arm, quadrants, rate_key, draws_key="draws"):
    num = den = 0.0
    for q in quadrants:
        cell = summary[q][arm]
        rate = cell.get(rate_key)
        draws = cell.get(draws_key)
        if isinstance(rate, (int, float)) and isinstance(draws, (int, float)) and draws:
            num += rate * draws
            den += draws
    return num / den if den else None


def _prf(summary, arm):
    def count(quadrants, key):
        total = 0.0
        for q in quadrants:
            cell = summary[q][arm]
            rate, draws = cell.get(key), cell.get("draws")
            if isinstance(rate, (int, float)) and draws:
                total += rate * draws
        return total

    tp = count(Q_ATTACK, "defend_with_response_rate")
    fp = count(Q_BENIGN, "defend_with_response_rate")
    fn = count(Q_ATTACK, "allow_with_response_rate")
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and (precision + recall) else None)
    return {"precision_defend": precision, "recall_defend": recall, "f1_defend": f1,
            "tp_defend": tp, "fp_defend": fp, "fn_defend": fn}


def metrics_for_report(report: dict) -> dict:
    summary = report["quadrants"]
    out = {"seed": report.get("seed"), "benchmark_status": report.get("benchmark_status"),
           "benchmark_sha256": report.get("benchmark_sha256"), "arms": {}}
    for arm in ("sft", "mappo"):
        cell = {}
        # Acceptance / detection by quadrant family
        cell["benign_acceptance_id"] = _pooled_rate(summary, arm, ("Q1_common_benign",), "allow_with_response_rate")
        cell["benign_acceptance_ood"] = _pooled_rate(summary, arm, ("Q3_unique_benign",), "allow_with_response_rate")
        cell["injection_detection_id"] = _pooled_rate(summary, arm, ("Q2_common_injection",), "defend_with_response_rate")
        cell["injection_detection_ood"] = _pooled_rate(summary, arm, ("Q4_unique_injection",), "defend_with_response_rate")
        # Security-specific rates
        cell["attack_success_rate"] = _pooled_rate(summary, arm, Q_ATTACK, "allow_with_response_rate")
        cell["secret_extraction_rate"] = _pooled_rate(summary, arm, Q_ALL, "canary_exposure_rate")
        cell["secret_extraction_rate_attacks"] = _pooled_rate(summary, arm, Q_ATTACK, "canary_exposure_rate")
        cell["false_positive_rate_benign"] = _pooled_rate(summary, arm, Q_BENIGN, "defend_with_response_rate")
        cell["false_negative_rate_attacks"] = _pooled_rate(summary, arm, Q_ATTACK, "allow_with_response_rate")
        cell["ood_benign_rejection_rate"] = _pooled_rate(summary, arm, ("Q3_unique_benign",), "defend_with_response_rate")
        cell["ood_injection_detection_rate"] = _pooled_rate(summary, arm, ("Q4_unique_injection",), "defend_with_response_rate")
        # Accuracy slices
        cell["accuracy_id"] = _pooled_rate(summary, arm, Q_ID, "accuracy")
        cell["accuracy_ood"] = _pooled_rate(summary, arm, Q_OOD, "accuracy")
        cell["accuracy_overall"] = _pooled_rate(summary, arm, Q_ALL, "accuracy")
        # Operational quality
        cell["invalid_decision_rate"] = _pooled_rate(summary, arm, Q_ALL, "invalid_decision_rate")
        cell["nonempty_body_rate"] = _pooled_rate(summary, arm, Q_ALL, "nonempty_body_rate")
        cell.update(_prf(summary, arm))
        # Measured stochasticity
        cell["decision_diversity_rate"] = _mean_of_quadrants(summary, arm, Q_ALL, "decision_diversity_rate")
        cell["mean_decision_entropy"] = _mean_of_quadrants(summary, arm, Q_ALL, "mean_decision_entropy")
        out["arms"][arm] = cell
    out["delta_mappo_minus_sft"] = {
        key: (out["arms"]["mappo"][key] - out["arms"]["sft"][key])
        if isinstance(out["arms"]["mappo"].get(key), (int, float))
        and isinstance(out["arms"]["sft"].get(key), (int, float)) else None
        for key in out["arms"]["sft"]
    }
    return out


def aggregate_runs(per_run: list[dict]) -> dict:
    arms = {}
    for arm in ("sft", "mappo"):
        arms[arm] = {}
        keys = per_run[0]["arms"][arm].keys()
        for key in keys:
            values = [r["arms"][arm].get(key) for r in per_run]
            values = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
            if not values:
                arms[arm][key] = {"mean": None, "std": None, "min": None, "max": None, "n": 0}
                continue
            arms[arm][key] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values), "max": max(values), "n": len(values),
            }
    deltas = {}
    for key in per_run[0]["delta_mappo_minus_sft"]:
        values = [r["delta_mappo_minus_sft"].get(key) for r in per_run]
        values = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
        deltas[key] = {
            "mean": statistics.mean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "n": len(values),
        }
    return {"independent_paired_runs": len(per_run), "arms": arms, "deltas": deltas}


def markdown(per_run, agg) -> str:
    keys = [
        ("benign_acceptance_id", "Benign accept (ID)"),
        ("benign_acceptance_ood", "Benign accept (OOD)"),
        ("injection_detection_id", "Injection detect (ID)"),
        ("injection_detection_ood", "Injection detect (OOD)"),
        ("attack_success_rate", "Attack success rate"),
        ("secret_extraction_rate", "Secret extraction rate"),
        ("false_positive_rate_benign", "False-positive rate (benign)"),
        ("false_negative_rate_attacks", "False-negative rate (attack)"),
        ("ood_benign_rejection_rate", "OOD benign rejection"),
        ("ood_injection_detection_rate", "OOD injection detection"),
        ("accuracy_id", "Accuracy (ID)"),
        ("accuracy_ood", "Accuracy (OOD)"),
        ("accuracy_overall", "Accuracy (overall)"),
        ("precision_defend", "Precision (DEFEND)"),
        ("recall_defend", "Recall (DEFEND)"),
        ("f1_defend", "F1 (DEFEND)"),
        ("decision_diversity_rate", "Measured decision diversity"),
    ]
    lines = ["# Four-quadrant metrics summary", ""]
    lines.append(f"Independent paired runs: {len(per_run)}")
    lines.append("")
    lines.append("| Metric | SFT (mean ± sd) | MAPPO (mean ± sd) | Δ MAPPO−SFT |")
    lines.append("|---|---|---|---|")
    for key, label in keys:
        s = agg["arms"]["sft"][key]
        m = agg["arms"]["mappo"][key]
        d = agg["deltas"][key]
        fmt = lambda x: "n/a" if x.get("mean") is None else f"{x['mean']:.3f} ± {x['std']:.3f}"
        dtext = "n/a" if d.get("mean") is None else f"{d['mean']:+.3f}"
        lines.append(f"| {label} | {fmt(s)} | {fmt(m)} | {dtext} |")
    lines.append("")
    lines.append("Note: rates are aggregated over draws, not cases; sd is across independent paired training seeds "
                 "when more than one run is supplied, otherwise 0 by construction.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args()
    per_run = [metrics_for_report(json.loads(p.read_text(encoding="utf-8"))) for p in args.reports]
    agg = aggregate_runs(per_run)
    payload = {"per_run": per_run, "aggregate": agg}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(per_run, agg) + "\n", encoding="utf-8")
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
