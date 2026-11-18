# Project State

## Current phase

One frozen confirmatory seed (42) is complete. The retained SFT adapter is at `checkpoints/sft_seed42_strong/` (10 epochs, 1e-4, best validation accuracy 0.980) and the paired MAPPO adapter at `checkpoints/mappo_seed42_strong/` (256 episodes, 16 PPO updates, best validation accuracy 0.939). The four-quadrant evaluation used a 48-case frozen benchmark and a runtime-only canary; artifacts are `reports/llm_eval_seed42.json` and `reports/full_metrics_seed42.md`. Full audit, methodology, results, limitations, and verdict are in `reports/audit_and_results.md`.

The seed-42 result does not establish general MAPPO superiority: OOD injection detection improved by 6.7 points and attack success rate fell by 4.2 points, but overall accuracy was unchanged and benign acceptance fell. The pre-registered four-quadrant pattern was not met by either arm. A second seed, matched compute, and benign training cases with adversarial untrusted content are needed before a firm conclusion.

The DistilBERT classifier experiment remains a separate track. Historical comparative claims were withdrawn because the reports conflict and the resampling analysis did not support the reported significance tests. The original exploratory data and benchmark remain in the repository but are not used for confirmatory claims. See `docs/llm_mappo_training.md` and `docs/four_quadrant_experiment_protocol.md` for training and evaluation procedures.

The historical Review 1–3 fixture milestones below document prior development and are not evidence that the current DistilBERT comparison has been trained or validated.

## Completed

- Three-review MAPPO-Sec architecture documented.
- Member 2 Defender master plan documented.
- **Review 1 complete (Parts 1–3, 2026-08-26):** Gate GATE-001–009, SFT corpus 20/8/10, scripts, 5 sanitized traces, 43 tests.
- **Part 4 Review 2 integration contracts (2026-08-26):** model_adapter, rollout_adapter, rewards. 104 tests.
- **Part 5 MAPPO update + capability retention (2026-08-27):** mappo_defender, capability_retention. 162 tests.
- **Part 6 Fixture episode harness (2026-08-27):** episode_harness, run_review2_episode. 208 tests.
- **Part 7 Review 2 docs (2026-08-27):** review2_integration_summary, defender-review2-evidence. 208 tests.
- **Part 8 Checkpoint evaluator (2026-08-27):** evaluate_checkpoint, heldout guard. 242 tests.
- **Part 9 Ablation configs + retention analysis (2026-08-28):** 3 ablation JSONs, retention script, JSON+CSV reports. 284 tests.
- **Part 10 Review 3 final artefacts (2026-08-28):**
  - `reports/review3_readiness_summary.json`: machine-readable readiness record — all 12 deliverables (READY or BLOCKED_EXTERNAL), fixture baseline metrics (in-dist + OOD + retention), full ablation plan with hypotheses, interface readiness (Member 1/3/4 PENDING), 7-step run instructions for when training is unblocked.
  - `docs/defender-review3-evidence.md`: full Review 3 presentation — 5-slice evaluation matrix, metric comparison tables (SFT/MAPPO BLOCKED_EXTERNAL), per-family ASR, capability retention table, 8-probe per-category rates, ablation plan with no-gate NOT_RUN rationale, end-to-end run instructions, interface readiness, 284-test coverage table, 9 generated reports indexed, 6 open blockers.
  - All 284 tests still pass.
- **Review 2 MAPPO Co-evolution Training Pipeline (2026-09-18):**
  - Resolved 10 runtime and interface bugs in `scripts/run_mappo_coevolution.py`.
  - Integrated `CentralizedCritic`, `AttackerPolicy`, `CoevolutionEnv`, and `CoevolutionWandbLogger`.
  - Added periodic and final checkpointing into `checkpoints/mappo_run`.
  - Integrated `ATTACK_CATEGORIES` and `ATTACK_TEMPLATES` from `src/attacker_policy/attack_generator.py`.
  - Added `log_metrics()` method to `CoevolutionWandbLogger`.
  - 430/430 tests passing. Full 1-episode smoke test verified successfully.

## In progress

Seed 42 is complete and documented. No training is in progress. Training runs are handled outside this workstream; request approval before starting any additional seed.

## Next exact action

Run a second seed with the same frozen splits and hyperparameters, preserving its manifests and checkpoints, then aggregate both reports with `scripts/analyze_llm_four_quadrant_report.py` to estimate variability. Before that, consider adding benign training cases with adversarial untrusted content so that over-refusal can be measured fairly. Keep the canary restricted to the evaluation process. Do not use the original inspected benchmark as confirmatory evidence.

## Latest verification

- Earlier check, before the synthetic-canary reward change: 359 `unittest` cases passed, including the decision-to-gate fail-closed path. Two pytest-dependent modules were skipped because `pytest` is unavailable in this environment; the full pytest suite was not run.
- Current checks: full `unittest` discovery ran 379 entries: 377 passed and two pytest-dependent modules errored at import because `pytest` is not installed. The focused data-preflight, four-quadrant hypothesis, training-context, and policy-gate suites pass (7, 5, 6, and 9 tests). Nineteen changed Python files parse; a synthetic two-seed aggregation check passed; `git diff --check` is clean. The checked-in data correctly fails confirmatory preflight because benign task-family annotations are absent; it remains exploratory.
- Model training and generation were not run because `peft` is unavailable here. The historical adapter under `defender_checkpoints/` is not a reproducible SFT/MAPPO pair.
- Historical repository record: `PYTHONPATH=. python3 -m pytest` reported 430 tests passed before the current paper/evaluation revisions.
- Historical MAPPO smoke test and dry-run checks are recorded below; they do not establish validity of the current comparative experiment.
- Scripts: the following historical checks were recorded:
  - `prepare_sft.py --dry-run`
  - `run_review1_baseline.py --dry-run`
  - `run_review2_episode.py --dry-run`
  - `evaluate_checkpoint.py --dry-run`
  - `run_capability_retention_analysis.py --dry-run`
  - `evaluate_checkpoint.py --checkpoint fixture_baseline --allow-heldout` (full run)

## Legacy classifier-track decisions

The decisions and blockers below refer to the earlier Review 1–3 / DistilBERT work. They do not describe the current causal-LM experiment.

- Base model: PENDING. SFT training BLOCKED_EXTERNAL.
- Dataset version: `review1-fixture-v2` / `sft-v2`.
- Heldout splits FROZEN — `allow_heldout=True` required.
- Gate: GATE-001–009 remain fixed; GATE-010 fails closed on invalid/DEFEND model decisions. The gate cannot be disabled for ablations.
- Action schema version: `gate-v1`.
- Reward weights: lambda_violation=1.0, lambda_refusal=0.3, lambda_cost=0.0.
- MAPPO defaults: clip_epsilon=0.2, beta_kl=0.1, value_loss_coeff=0.5, entropy_coeff=0.01, gamma=0.99, gae_lambda=0.95.
- Ablation betas: no_kl=0.0, default=0.1, high_kl=1.0.

## Blockers / questions for team

- Base LLM/checkpoint, training stack, LoRA/QLoRA/full-SFT, GPU budget → SFT BLOCKED_EXTERNAL.
- Member 1 AttackPayload schema → MAPPO self-play blocked.
- Member 3 Trajectory/Critic schema → trajectory assembly blocked.
- Member 4 EvaluationEvent/OOD suite → Review 3 eval blocked.
