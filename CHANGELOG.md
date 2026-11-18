# Changelog

## 2026-11-19

- Completed the frozen confirmatory experiment for seed 42: SFT baseline, MAPPO adaptation, and paired four-quadrant evaluation. Results are recorded in `reports/audit_and_results.md`, `reports/llm_eval_seed42.json`, and `reports/full_metrics_seed42.md`.
- Corrected the legacy SFT schema to accept the `task_family` annotation, restoring 456 passing tests.
- Added a shared generation configuration, measured decision stochasticity, logged entropy coefficient and bandit-advantage scope in the MAPPO manifest, and aligned the end-to-end reproduction script with the retained checkpoints.
- Added an opt-in `--reference-dtype float16` option for memory-constrained training; the default remains float32 to match the reported run.
- Final verdict: one seed shows a narrow OOD attack-detection gain for MAPPO with a benign-usability cost and identical overall accuracy; the general superiority hypothesis is not supported.

## 2026-11-18

- Reframed the phase 3 report as a preliminary implementation and evidence audit; removed unsupported comparative performance and significance claims.
- Documented a four-quadrant confirmatory protocol with a new sealed test set, per-case reporting, fixed decision criteria, multiple training seeds, and explicit checkpoint/data provenance.
- Updated the classifier trainers and evaluators to separate validation from held-out evaluation, reject cross-split ID/scenario/exact-input duplicates, require trained checkpoints, record run provenance, and avoid treating resamples as independent training runs.
- Rewrote Sections 3 and 4 to describe the repository’s DistilBERT classifier pipelines and distinguish them from the separate Qwen smoke test.
- Marked prior significance and four-quadrant outputs as unverified exploratory artifacts; removed unsupported benchmark claims from the demo interfaces.

## 2026-08-26 - Initial Defender Review 1 foundation

- Added strict tool policy gate, isolated fixture tools, inert sandbox, and audit logging.
- Added SFT starter corpus, training configuration, baseline evaluator, and reports.
- Added architecture, master plan, and persistent project-state workflow.
- Verified 10 Defender tests pass.

## 2026-08-26 - Part 1 gate and sandbox hardening

- `gate.py`: Added `PolicyDecision` dataclass (rule_id, reason_code, safe_message). Added GATE-004 (oversized purpose), GATE-005 (secret-access patterns), GATE-006 (external-URL/code-exec), GATE-009 (safe tool-failure denial). GateResult now carries an optional PolicyDecision on every outcome.
- `audit.py`: Added `_redact()` scrubbing email addresses, IPv4 addresses, JWT/bearer tokens, API-key shapes (≥32 chars), and credential key=value pairs before AuditEvent storage.
- `evaluation.py`: Added `tool_block_rate`, `unsafe_proposal_rate` (pre-gate), and `per_family_asr` metrics.
- `tests/test_gate_hardening.py`: 28 new tests covering the full Part 1 test matrix (oversized input, secret-access, external-URL/code-exec, indirect injection, tool failure, PolicyDecision fields, audit redaction, malformed calls).
- All 38 tests pass.

## 2026-08-26 - Part 2 SFT corpus expansion and split discipline

- `data/sft/train.jsonl`: 12 → 20 examples; added indirect injection (4 families/sources), over-refusal controls, tool-confusion; added 10 required schema fields.
- `data/sft/validation.jsonl`: 4 → 8 examples; added indirect ticket injection, CVE over-refusal, cross-scope tool-confusion, sandbox classification case.
- `data/sft/heldout.jsonl`: new 10-example SFT held-out corpus with disjoint scenario_ids; frozen.
- `data/evaluation/review1_cases.jsonl`: 8 → 15 cases; added indirect log/tool-result injection, two tool-confusion cases with invalid_tool_attempted metadata.
- `data/evaluation/heldout_cases.jsonl`: new 10-case evaluation held-out; frozen.
- `src/defender_policy/sft_data.py`: upgraded to 10-field schema; source_type/attack_family enums; label-family consistency; three-way split (ID + scenario_id disjointness).
- `src/defender_policy/baseline.py`: extended INJECTION_SIGNALS (indirect patterns) and UNSAFE_REQUESTS (export_data, path-traversal, api keys, delete-all); sandbox_text trigger.
- `tests/test_review1_pipeline.py`: updated counts, added 6 new tests (three-way split, attack families, source types, per-family ASR, tool-block rate).
- All 43 tests pass.

## 2026-08-26 - Part 3 Review 1 evaluation and presentation pack

- `scripts/prepare_sft.py`: added --config, --output-dir, --dry-run flags; three-way split validation; run metadata annotation.
- `scripts/run_review1_baseline.py`: added --config, --seed, --output-dir, --dry-run flags; produces richer metrics (per_family_asr, tool_block_rate, run_metadata, summary_table with SFT column BLOCKED_EXTERNAL); writes sanitized traces report.
- `configs/evaluation.json`: created with fixed seed, paths, metric list, gate_version, and heldout_policy.
- `reports/review1_baseline_metrics.json`: regenerated — ASR=0.0, benign_success=1.0, false_refusal=0.0, tool_block_rate=1.0, all per-family ASRs=0.0.
- `reports/sft_warmup_manifest.json`: regenerated — train=20, val=8, heldout=10; all splits clean.
- `reports/review1_traces.json`: created — 5 sanitized traces (T1 safe answer, T2 safe tool use, T3 direct injection block, T4 indirect injection resistance, T5 safe refusal); no raw injection content stored.
- Both scripts exit 0. All 43 tests pass.
- Review 1 is complete. SFT training remains BLOCKED_EXTERNAL pending team base-model decision.

## 2026-08-26 - Part 4 Review 2 integration contracts

- `src/defender_policy/model_adapter.py`: `DefenderObservation` (frozen, no critic state), `RawModelOutput` (pre-gate proposal with log_prob/entropy), abstract `BaseModelAdapter` with validated `act()` contract, `FixtureModelAdapter` (wraps ReviewOneBaselineDefender), `SFTModelAdapter` stub (BLOCKED_EXTERNAL, NotImplementedError).
- `src/defender_policy/rollout_adapter.py`: `DefenderAction` (frozen; carries tool_call, gate_result, pre_gate_unsafe flag, log_prob), `TrajectoryStep` (observation+action+reward+components; advantage/value_estimate reserved for Member 3), `_is_pre_gate_unsafe()` heuristic, `RolloutAdapter.step()` (routes every tool proposal through unchanged gate).
- `src/defender_policy/rewards.py`: `RewardWeights` (non-negative validated), `task_utility()`, `safety_violation()` (additive, capped 1.0), `unnecessary_refusal()` (conservative), `excessive_cost()` (0.0 v1), `compute_reward()`, `apply_reward_to_step()`. KL term excluded — belongs in mappo_defender.py.
- `tests/test_review2_adapters.py`: 61 new tests across 11 test classes.
- All 104 tests pass.

## 2026-08-27 - Part 5 MAPPO Defender update stub and capability retention

- `src/defender_policy/mappo_defender.py`: Pure-Python, framework-agnostic MAPPO loss. `MAPPOConfig` (clip_epsilon∈(0,1], beta_kl≥0, gamma/gae_lambda∈[0,1]). `compute_kl_penalty` (zero when current==ref; monotone in beta_kl). `compute_policy_loss` (PPO clipped surrogate; correct clip for negative advantages). `compute_value_loss` (MSE). `compute_entropy_bonus`. `compute_total_loss → MAPPOLossOutput`. `defender_update_step` (full-minibatch convenience wrapper). KL lives here; gate not touched.
- `src/defender_policy/capability_retention.py`: 8 built-in `FIXTURE_PROBES` (incident_summary, security_vocabulary, ioc_extraction, tool_use, log_analysis, policy_explanation). `run_capability_probe()` runner with per-category pass rates and regression flag. Gate active during probing.
- `tests/test_mappo_defender.py`: 58 new tests (MAPPOConfig, KL penalty, policy loss, value loss, entropy bonus, total loss, defender_update_step, capability retention).
- All 162 tests pass.

## 2026-08-27 - Part 6 Review 2 fixture episode harness

- `src/defender_policy/episode_harness.py`: `EpisodeResult` dataclass (steps, total_reward, reward_components, gate_blocked, pre_gate_unsafe, retention_report, scope disclaimer). `FixtureEpisodeHarness` with run_episode(), run_episode_for_case() (deterministic), run_n_episodes(n). Single-turn pipeline: FixtureModelAdapter → RolloutAdapter.step() → apply_reward_to_step() → run_capability_probe(). Gate never bypassed. Advantage/value left None (Member 3 only).
- `scripts/run_review2_episode.py`: CLI --n-episodes, --seed, --output-dir, --dry-run. Aggregates avg_reward, gate_block_count, pre_gate_unsafe_count, per_family_avg_reward.
- `reports/review2_episode_report.json`: generated — 14 episodes, avg_reward=0.2857, gate_blocks=0, retention_score=1.0.
- `tests/test_episode_harness.py`: 46 new tests (construction, run_episode, run_episode_for_case, reward integration, capability retention, run_n_episodes, aggregation helpers, eval expected mapping).
- All 208 tests pass.

## 2026-08-27 - Part 7 Review 2 integration report and documentation

- `configs/sft_warmup.json`: added heldout_file, mappo_config (clip_epsilon, beta_kl, value_loss_coeff, entropy_coeff, gamma, gae_lambda, max_grad_norm + gate-fixed note), reward_weights (lambda values), capability_retention spec, sft_status=BLOCKED_EXTERNAL.
- `reports/review2_integration_summary.json`: consolidated deliverables — four parts, fixture baseline metrics, interface readiness (DefenderAction READY; Members 1/3/4 PENDING), open blockers, next steps.
- `docs/defender-review2-evidence.md`: full presentation evidence — architecture diagram, interface tables, gate rule table, reward/MAPPO formulas, SFT corpus table, metrics table, capability probe table, episode stats, sanitized traces, test coverage (208), blockers, files index.
- All 208 tests pass; all scripts exit 0 (dry-run verified).
- Review 2 fixture implementation complete. SFT+MAPPO training BLOCKED_EXTERNAL.

## 2026-08-27 - Part 8 Review 3 checkpoint evaluation harness

- `src/defender_policy/evaluate_checkpoint.py`: `SliceMetrics` (frozen; all standard metrics + avg_total_reward + evaluation_scope), `CheckpointEvalResult` (in_distribution, heldout_ood gated by allow_heldout), `_AdapterShim` (routes adapter+rollout through evaluate_cases), `evaluate_checkpoint()` with allow_heldout guard (default False).
- `scripts/evaluate_checkpoint.py`: CLI --checkpoint, --allow-heldout, --seed, --output-dir, --dry-run.
- `reports/checkpoint_eval_fixture_baseline.json`: generated — in-dist ASR=0.0, tool_block=1.0, avg_reward=0.857; OOD ASR=0.0, avg_reward=0.90.
- `tests/test_evaluate_checkpoint.py`: 34 new tests (basic contract, in-dist metrics, heldout guard, reproducibility, custom id).
- All 242 tests pass.

## 2026-08-28 - Part 9 Ablation configs and capability-retention analysis

- `configs/ablation_configs/ablation_no_kl.json`: beta_kl=0.0; threshold=0.875; gate ACTIVE; description and hypothesis for Review 3.
- `configs/ablation_configs/ablation_high_kl.json`: beta_kl=1.0; threshold=1.0; gate ACTIVE; description and hypothesis.
- `configs/ablation_configs/ablation_no_gate.json`: documents WHY gate cannot be disabled; feasible alternative; Review 3 table entry as NOT_RUN.
- `scripts/run_capability_retention_analysis.py`: CLI --checkpoint, --threshold, --output-dir, --dry-run. Writes JSON (full per-probe) and CSV (for plots). Console output: per-category ✓/△/✗ table and per-probe pass/fail.
- `reports/capability_retention_fixture_baseline.json`: score=1.0, 8/8 probes pass, 6 categories all 1.0.
- `reports/capability_retention_fixture_baseline.csv`: per-probe CSV for plotting.
- `tests/test_ablation_configs.py`: 42 new tests (file existence, JSON validity, no_kl/high_kl/no_gate schema, MAPPOConfig compatibility, retention analysis, build_report structure, script smoke).
- All 284 tests pass.

## 2026-08-28 - Part 10 Review 3 final summary and presentation artefacts

- `reports/review3_readiness_summary.json`: machine-readable readiness record — all deliverables (READY or BLOCKED_EXTERNAL), fixture metrics, ablation plan with hypotheses, interface readiness, 7-step run instructions.
- `docs/defender-review3-evidence.md`: full Review 3 presentation — 5-slice evaluation matrix, metric tables (BLOCKED_EXTERNAL placeholders), per-family ASR, capability retention table (8 probes, 6 categories), ablation plan with no-gate NOT_RUN rationale, end-to-end instructions, interface readiness, 284-test coverage, 9 reports indexed, 6 blockers.
- All 284 tests pass. All scripts exit 0. All three review phases complete as fixture infrastructure.
- Project in HOLDING state pending base model confirmation and Member 1/3/4 schemas.

## 2026-09-18 - Review 2 MAPPO Co-evolution Training Pipeline & Integration

- `scripts/run_mappo_coevolution.py`:
  - Resolved 10 runtime bugs (observation tensor encoding via `encode_obs()`, unpacking `AttackDecision`, unified payload mapping, `EnvStepResult` rewards, `CentralizedCritic` config-driven `critic_lr`, `ppo_epochs` training loop, `CTDETrajectoryBuffer` tensor storage).
  - Integrated `ATTACK_CATEGORIES` and `ATTACK_TEMPLATES` from `src/attacker_policy/attack_generator.py`.
  - Added periodic and final model checkpointing for attacker and centralized critic.
  - Added step-level and episode-level metric logging via `CoevolutionWandbLogger`.
  - Safe CUDA device selection with automatic CPU fallback.
- `configs/mappo_config.yaml`:
  - Configured 500 episodes, 6 steps per episode, tuned critic lr (`0.00002`), max grad norm (`0.5`), explicit `checkpoint_dir` (`checkpoints/mappo_run`), `wandb_mode: disabled`, and `output_dir: reports`.
- `src/evaluation_metrics/wandb_logger.py`:
  - Added `log_metrics()` method to safely record arbitrary scalar dictionaries to W&B tracker.
- `tests/test_mappo_attacker.py`:
  - Added unit tests for `encode_obs`, `get_attack_payload`, and `CoevolutionWandbLogger.log_metrics()`.
  - All 430 tests passing.
