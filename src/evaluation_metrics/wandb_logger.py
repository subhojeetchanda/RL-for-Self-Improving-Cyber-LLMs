"""evaluation_metrics/wandb_logger.py — Member 4: Coevolution W&B logging.

Responsibilities
----------------
1. Log per-step MAPPO coevolution metrics to Weights & Biases:
       asr, defender_reward, attacker_reward, critic_loss,
       kl_divergence, gate_block_rate, pre_gate_unsafe_rate
2. Log per-episode summary scalars.
3. Write ``review2_coevolution_metrics.json`` (ASR vs. training step).
4. Write ``review2_adversarial_traces.json`` (successful vs. defended samples).

Design rules
------------
- Built on top of the existing ``WandbTracker``; adds coevolution-specific
  logging methods without modifying the base class.
- Gracefully degrades when wandb is unavailable: metrics are accumulated
  in memory and the JSON reports are written to disk only.
- No model calls, no I/O outside the configured output directory.
- All JSON outputs are human-readable (indent=2) so teammates can inspect
  them without running the full pipeline.
- Successful attack traces are logged with a ``[REDACTED]`` marker for any
  field that would expose real attacker payloads to the review document.

Usage
-----
    from evaluation_metrics.wandb_logger import CoevolutionWandbLogger

    logger = CoevolutionWandbLogger(output_dir="reports/", run_name="r2_coevo")
    logger.init(tags=["review2", "coevolution"])

    for step_result in episode_steps:
        logger.log_step(step_result, global_step=step_idx)

    logger.log_episode(episode_result)
    logger.flush_reports()   # writes both JSON files to output_dir
    logger.finish()
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .tracking_utils import WandbTracker, log_scalar_dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public metric names (used as dict keys and W&B panel names)
# ---------------------------------------------------------------------------

METRIC_ASR = "attack_success_rate"
METRIC_DEF_REWARD = "defender_reward"
METRIC_ATT_REWARD = "attacker_reward"
METRIC_CRITIC_LOSS = "critic_loss"
METRIC_KL_DIV = "kl_divergence"
METRIC_GATE_BLOCK_RATE = "gate_block_rate"
METRIC_PRE_GATE_UNSAFE_RATE = "pre_gate_unsafe_rate"
METRIC_EPISODE_RETURN_DEF = "episode_return_defender"
METRIC_EPISODE_RETURN_ATT = "episode_return_attacker"
METRIC_EPISODE_ASR = "episode_attack_success_rate"


# ---------------------------------------------------------------------------
# CoevolutionWandbLogger
# ---------------------------------------------------------------------------

class CoevolutionWandbLogger:
    """Logging facade for the Review 2 MAPPO coevolution training loop.

    Parameters
    ----------
    output_dir : str | Path
        Directory where JSON report files are written.  Created if absent.
    run_name : str, optional
        Human-readable W&B run name.
    config_path : str | Path
        Path to ``configs/wandb_config.yaml``.
    mode : str, optional
        W&B mode: ``"online"``, ``"offline"``, or ``"disabled"``.
        Defaults to ``"disabled"`` so unit tests never touch the network.
    """

    COEVO_METRICS_FILENAME = "review2_coevolution_metrics.json"
    ADV_TRACES_FILENAME = "review2_adversarial_traces.json"

    def __init__(
        self,
        output_dir: str | Path = "reports",
        run_name: Optional[str] = None,
        config_path: str | Path = "configs/wandb_config.yaml",
        mode: Optional[str] = None,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._tracker = WandbTracker(
            config_path=config_path,
            run_name=run_name or f"r2_coevo_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}",
            mode=mode,
        )

        # Accumulated step-level metric rows for the coevolution metrics report
        self._step_rows: List[Dict[str, Any]] = []
        # Accumulated traces for successful attacks and clean defences
        self._successful_attack_traces: List[Dict[str, Any]] = []
        self._defended_traces: List[Dict[str, Any]] = []
        # Episode-level summary rows
        self._episode_rows: List[Dict[str, Any]] = []
        self._global_step: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init(
        self,
        tags: Optional[List[str]] = None,
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> "CoevolutionWandbLogger":
        """Initialize the underlying W&B run.  Safe to call if wandb absent."""
        self._tracker.init(
            tags=tags or ["review2", "coevolution"],
            extra_config=extra_config,
        )
        return self

    def finish(self) -> None:
        """Finalize the W&B run and ensure reports are flushed to disk."""
        self.flush_reports()
        self._tracker.finish()

    # ------------------------------------------------------------------
    # Per-step logging
    # ------------------------------------------------------------------

    def log_step(
        self,
        step_result: Any,            # EnvStepResult from coevolution_env
        global_step: Optional[int] = None,
        critic_loss: Optional[float] = None,
        kl_divergence: Optional[float] = None,
    ) -> None:
        """Log one environment step's metrics.

        Parameters
        ----------
        step_result : EnvStepResult
            Result returned by ``CoevolutionEnv.step()``.
        global_step : int, optional
            Global training step counter.  Auto-increments if None.
        critic_loss : float, optional
            Value-network loss for this step (provided by Member 3's critic).
            Logged as 0.0 when not available.
        kl_divergence : float, optional
            KL divergence from SFT reference policy (from mappo_defender.py).
            Logged as 0.0 when not available.
        """
        if global_step is not None:
            self._global_step = global_step
        else:
            self._global_step += 1

        info = step_result.info
        injection_detected = info.get("injection_obedience_detected", False)
        gate_blocked = info.get("gate_blocked", False)
        pre_gate_unsafe = info.get("pre_gate_unsafe", False)

        metrics = {
            METRIC_ASR: 1.0 if injection_detected else 0.0,
            METRIC_DEF_REWARD: step_result.reward_defender,
            METRIC_ATT_REWARD: step_result.reward_attacker,
            METRIC_CRITIC_LOSS: critic_loss if critic_loss is not None else 0.0,
            METRIC_KL_DIV: kl_divergence if kl_divergence is not None else 0.0,
            METRIC_GATE_BLOCK_RATE: 1.0 if gate_blocked else 0.0,
            METRIC_PRE_GATE_UNSAFE_RATE: 1.0 if pre_gate_unsafe else 0.0,
        }

        # Log to W&B
        log_scalar_dict(self._tracker, metrics, step=self._global_step)

        # Accumulate for JSON report
        row: Dict[str, Any] = {
            "global_step": self._global_step,
            "episode_id": info.get("case_id", "unknown"),
            "step_index": info.get("step_index", -1),
            "case_kind": info.get("case_kind", "unknown"),
            "attack_family": info.get("attack_family"),
            **metrics,
        }
        self._step_rows.append(row)

        # Accumulate traces
        snippet = info.get("defender_answer_snippet", "")
        gs = step_result.global_state
        attacker_payload = (
            gs.attacker_text if hasattr(gs, "attacker_text") else ""
        )

        if injection_detected and info.get("case_kind") == "attack":
            self._successful_attack_traces.append({
                "global_step": self._global_step,
                "case_id": info.get("case_id"),
                "attack_family": info.get("attack_family"),
                "attacker_payload_snippet": attacker_payload[:80] if attacker_payload else "[REDACTED]",
                "defender_answer_snippet": snippet,
                "reward_defender": step_result.reward_defender,
                "outcome": "attacker_succeeded",
            })
        elif not injection_detected and not gate_blocked and info.get("case_kind") == "attack":
            self._defended_traces.append({
                "global_step": self._global_step,
                "case_id": info.get("case_id"),
                "attack_family": info.get("attack_family"),
                "defender_answer_snippet": snippet,
                "reward_defender": step_result.reward_defender,
                "outcome": "defender_resisted",
            })

    # ------------------------------------------------------------------
    # Per-episode logging
    # ------------------------------------------------------------------

    def log_episode(self, episode_result: Any) -> None:
        """Log a complete episode summary.

        Parameters
        ----------
        episode_result : CoevolutionEpisodeResult
            Result returned by ``CoevolutionEnv.run_episode()``.
        """
        metrics = {
            METRIC_EPISODE_RETURN_DEF: episode_result.total_defender_reward,
            METRIC_EPISODE_RETURN_ATT: episode_result.total_attacker_reward,
            METRIC_EPISODE_ASR: episode_result.attack_success_rate,
        }
        log_scalar_dict(self._tracker, metrics)

        self._episode_rows.append({
            "episode_id": episode_result.episode_id,
            "case_id": episode_result.case_id,
            "case_kind": episode_result.case_kind,
            "attack_family": episode_result.attack_family,
            "step_count": episode_result.step_count,
            "total_defender_reward": episode_result.total_defender_reward,
            "total_attacker_reward": episode_result.total_attacker_reward,
            "attack_success_rate": episode_result.attack_success_rate,
            "gate_block_count": episode_result.gate_block_count,
        })

    # ------------------------------------------------------------------
    # Bulk step logging (for post-hoc logging from script)
    # ------------------------------------------------------------------

    def log_steps_from_episode(
        self,
        episode_result: Any,
        base_step: int = 0,
        critic_loss: Optional[float] = None,
        kl_divergence: Optional[float] = None,
    ) -> int:
        """Log all steps from a ``CoevolutionEpisodeResult``.

        Returns the updated global_step counter.
        """
        for i, step_result in enumerate(episode_result.steps):
            self.log_step(
                step_result,
                global_step=base_step + i,
                critic_loss=critic_loss,
                kl_divergence=kl_divergence,
            )
        self.log_episode(episode_result)
        return base_step + len(episode_result.steps)

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def flush_reports(self) -> Dict[str, Path]:
        """Write accumulated metrics to disk as JSON reports.

        Returns
        -------
        dict
            Mapping of report name to written Path.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        written: Dict[str, Path] = {}

        # ── review2_coevolution_metrics.json ──────────────────────────
        n_steps = len(self._step_rows)
        n_eps = len(self._episode_rows)

        avg_asr = (
            sum(r[METRIC_ASR] for r in self._step_rows) / n_steps
            if n_steps > 0 else 0.0
        )
        avg_def_reward = (
            sum(r[METRIC_DEF_REWARD] for r in self._step_rows) / n_steps
            if n_steps > 0 else 0.0
        )
        avg_att_reward = (
            sum(r[METRIC_ATT_REWARD] for r in self._step_rows) / n_steps
            if n_steps > 0 else 0.0
        )
        avg_gate_block = (
            sum(r[METRIC_GATE_BLOCK_RATE] for r in self._step_rows) / n_steps
            if n_steps > 0 else 0.0
        )

        # Per-family ASR
        family_asr: Dict[str, List[float]] = {}
        for row in self._step_rows:
            fam = row.get("attack_family") or "benign"
            family_asr.setdefault(fam, []).append(row[METRIC_ASR])
        per_family_avg_asr = {
            fam: sum(vals) / len(vals)
            for fam, vals in family_asr.items()
            if vals
        }

        coevo_report: Dict[str, Any] = {
            "run_metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "scope": "fixture coevolution metrics; not a trained-LLM result",
                "n_steps": n_steps,
                "n_episodes": n_eps,
                "defender": "FixtureModelAdapter (ReviewOneBaselineDefender)",
                "attacker": "DummyAttacker (stub — real MAPPOAttacker pending M1)",
                "critic": "None (advantage computation pending M3)",
            },
            "aggregate": {
                "mean_attack_success_rate": round(avg_asr, 4),
                "mean_defender_reward": round(avg_def_reward, 4),
                "mean_attacker_reward": round(avg_att_reward, 4),
                "mean_gate_block_rate": round(avg_gate_block, 4),
                "per_family_mean_asr": {
                    k: round(v, 4) for k, v in per_family_avg_asr.items()
                },
            },
            "asr_vs_step": [
                {
                    "global_step": r["global_step"],
                    "asr": r[METRIC_ASR],
                    "defender_reward": r[METRIC_DEF_REWARD],
                    "attacker_reward": r[METRIC_ATT_REWARD],
                    "gate_block": r[METRIC_GATE_BLOCK_RATE],
                    "attack_family": r.get("attack_family"),
                    "case_kind": r.get("case_kind"),
                }
                for r in self._step_rows
            ],
            "episode_summaries": self._episode_rows,
            "disclaimer": (
                "All metrics are from a deterministic rule-based fixture baseline. "
                "They validate the coevolution environment only. "
                "Do not present as trained-LLM or production-security results."
            ),
        }

        metrics_path = self._output_dir / self.COEVO_METRICS_FILENAME
        metrics_path.write_text(json.dumps(coevo_report, indent=2), encoding="utf-8")
        written["coevo_metrics"] = metrics_path
        logger.info("Wrote coevolution metrics to %s", metrics_path)

        # Upload to W&B if active
        self._tracker.upload_artifact(metrics_path, artifact_type="metrics",
                                       name="review2_coevo_metrics")

        # ── review2_adversarial_traces.json ───────────────────────────
        traces_report: Dict[str, Any] = {
            "run_metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "scope": "fixture adversarial trace log; not a trained-LLM result",
                "n_successful_attack_steps": len(self._successful_attack_traces),
                "n_defended_steps": len(self._defended_traces),
            },
            "successful_attacks": self._successful_attack_traces,
            "defended_cases": self._defended_traces,
            "disclaimer": (
                "Attacker payload snippets are truncated to 80 characters and "
                "represent dummy-attacker strings, NOT real adversarial injections. "
                "Do not present as a red-team result."
            ),
        }

        traces_path = self._output_dir / self.ADV_TRACES_FILENAME
        traces_path.write_text(json.dumps(traces_report, indent=2), encoding="utf-8")
        written["adv_traces"] = traces_path
        logger.info("Wrote adversarial traces to %s", traces_path)

        self._tracker.upload_artifact(traces_path, artifact_type="metrics",
                                       name="review2_adv_traces")

        return written

    def __repr__(self) -> str:
        return (
            f"CoevolutionWandbLogger("
            f"steps={len(self._step_rows)}, "
            f"episodes={len(self._episode_rows)}, "
            f"tracker={self._tracker!r})"
        )
