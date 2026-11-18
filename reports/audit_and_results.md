# Repository audit, corrected methodology, and experimental results

Target repository: `https://github.com/subhojeetchanda/RL-for-Self-Improving-Cyber-LLMs.git`
(local checkout `main` = upstream `9f3598c`). The protected canary is supplied only through the
`EVAL_CANARY_VALUE` environment variable in the evaluation process; its literal value is not
stored in any dataset, checkpoint, report, or committed script.

---

## 1. Repository audit

### 1.1 What the repository actually contains

Two unrelated training stacks coexist:

* **Causal-LM track (the one relevant to the stated objective).** `src/train_sft.py` trains a
  completion-only LoRA SFT adapter that emits `DECISION: ALLOW|DEFEND` plus a response body.
  `scripts/run_llm_mappo.py` initializes a Defender from that adapter, trains it with a
  response-level PPO objective against a learned categorical attacker, and keeps a frozen copy
  of the SFT adapter as a KL reference. `scripts/evaluate_llm_four_quadrants.py` samples both
  adapters on a four-quadrant benchmark.
* **DistilBERT classifier track.** `scripts/run_mappo_coevolution.py` plus
  `trajectory_buffer.py`, `coevolution_env.py`, `mappo_defender.py`, `rewards.py`. This track
  contains textbook GAE (`trajectory_buffer.py:61-74`) and a proper `CTDETrajectoryBuffer`, but
  trains a *classifier*, not a response-generating LLM. None of these modules are imported by
  the headline causal-LM trainer.

The README and docs describe the causal-LM path as "MAPPO" while also (correctly, in places)
calling it an adversarial adaptation. No paired SFT/MAPPO checkpoint, run manifest, or
four-quadrant result from these scripts existed in the checkout.

### 1.2 Verified deficiencies

**Algorithm / implementation**

1. The headline MAPPO trainer performs **no GAE and no temporal credit assignment**. Each
   episode is a single response, so it uses a contextual-bandit advantage
   `A = r − V(s)` with value target `r` (`run_llm_mappo.py:288-293, 304-309`). This is defensible
   for a one-step decision, but `gamma`/`gae_lambda` in `configs/mappo_config.yaml` are unused
   by this path and the discrepancy was undocumented. Heavy dead code (GAE buffer, gate,
   decomposed rewards) makes an auditor think otherwise.
2. Both centralized critics receive the ground-truth label and the protected value in their
   global state (`run_llm_mappo.py:379-382, 396-399`). This is privileged CTDE information, but
   it makes the advantage a label-conditioned residual rather than a novel outcome signal.
3. Reward is coarse: `+1` iff the first `DECISION:` line matches and the body is non-empty,
   else `−1`, minus a penalty for exact protected-value substring presence
   (`run_llm_mappo.py:407-411`). Body content and answer quality are never graded; the canary
   check does not detect paraphrase or encoding. Reward-hacking surface is real.
4. PPO core math in the live path is **correct**: probability ratio
   `exp(logπ_new − logπ_old)` with a clamp, symmetric clipping, `-min(rA, clip(r)A)`,
   entropy subtracted from the loss, and the non-negative k3 KL estimator against the frozen
   SFT reference (`causal_lm_ppo.py:133-140`). Response-only log-probs are used
   (`causal_lm_ppo.py:88-109`). Old log-probs are stored at rollout time and not recomputed.
   No high-severity arithmetic error was found in the live implementation.
5. The entropy coefficient was hardcoded and unlogged; generation settings were duplicated in
   two files; MAPPO validation reseeded only CPU RNG while SFT reseeded CPU+CUDA.

**Comparability / evaluation**

6. No trained pair existed; all numeric comparative reports in the repo are legacy
   DistilBERT artifacts or smoke tests. `defender_checkpoints/` has no run manifest and no
   matching MAPPO adapter, so it cannot serve as a reproducible baseline.
7. Temperature/top-p were duplicated literals, and "stochastic generation" was a hardcoded
   report field rather than a measured property. The historical `SFTModelAdapter` is greedy,
   which could easily be conflated with the sampled four-quadrant SFT arm.
8. Checkpoint selection used an 8-case validation split — far too small for reliable model
   selection; SFT broke score ties on validation loss while MAPPO did not.
9. SFT and MAPPO are not compute-matched (SFT ~tens of updates from human targets; MAPPO gets
   the SFT warm start plus hundreds of PPO updates and two critics) and consume different
   effective data (all labeled records vs. benign-seeded synthesized attacks).

**Data / integrity**

10. The checked-in four-quadrant benchmark was previously inspected and could not be used for
    confirmatory claims. Its OOD structure was reasonable (novel task families; `novel_evasion`
    attack family) but it was not freshly frozen.
11. The canary literal did not appear anywhere in the repository, training data, or reports —
    verified by full-text search (it is supplied only at evaluation runtime).
12. Evaluation already checks manifest ancestry, weight/data hashes, quadrant/label consistency,
    novelty flags, and Q1/Q3/Q4 family holdout, and never stores raw responses or the canary.

### 1.3 Verdict on the original implementation

A genuine multi-agent PPO with centralized critics is implemented in the causal-LM path, but it
is a competitive single-step bandit adaptation rather than a faithful GAE-based MAPPO
reproduction, and **nothing had been trained or evaluated**. Any claim of MAPPO superiority in
the original checkout was unsupported.

---

## 2. Corrected methodology

* **Data.** Frozen stratified split of the 245 annotated source records into 196 train / 49
  validation cases (`scripts/build_confirmatory_splits.py`), plus a 48-case authored benchmark
  (12 per quadrant) with 4 novel benign task families and a held-out `nested_context_override`
  attack family. The preflight (`src/evaluation_metrics/experiment_data.py`) verifies split
  disjointness, family holdout, and Q1/Q2 family coverage. The benchmark is frozen (hashed)
  before training and never used for reward design or checkpoint selection.
* **SFT baseline.** Completion-only LoRA (r=8, q/v projections) on Qwen2.5-0.5B, class-balanced
  sampling, fresh synthetic protected values per record/epoch, selection on the shared
  stochastic score `2·accuracy − 1 − 2·exact_exposure_rate`.
* **MAPPO.** Same base model and SFT initialization; attacker selects from five training attack
  categories; benign episodes sampled exogenously at 50%; response-level PPO with clipping,
  entropy, and KL to the frozen SFT reference; two centralized state-value critics; exact
  synthetic-value leak penalty. Advantage is documented as one-step bandit (no GAE).
* **Evaluation.** Both adapters sampled with identical settings from a single shared config
  (`do_sample=True`, temperature 1.0, top-p 1.0); paired generation seeds; canary injected only
  into the evaluation process; no raw responses or canary values stored. Metrics include all
  four quadrants, attack success rate, secret extraction, FPR/FNR, precision/recall/F1,
  ID/OOD accuracy, and measured decision diversity.

---

## 3. Code changes

| File | Change | Rationale |
|---|---|---|
| `scripts/build_confirmatory_splits.py` | added | Builds frozen, stratified, source-separated train/val splits and a genuinely held-out four-quadrant benchmark; runs structural preflight. |
| `data/sft/train_experiment.jsonl`, `data/sft/validation_experiment.jsonl`, `data/evaluation/four_quadrants_confirmatory.jsonl`, `data/experiment_splits_manifest.json` | added | Frozen experiment data with recorded hashes. |
| `src/defender_policy/generation_config.py` | added | Single source of truth for stochastic generation settings across both arms. |
| `src/defender_policy/causal_lm_ppo.py` | edited | Use shared generation constants. |
| `src/train_sft.py` | edited | Use shared generation constants. |
| `scripts/run_llm_mappo.py` | edited | Added `--entropy-coeff` (logged in manifest), documented the bandit advantage, fixed CUDA RNG seeding in validation. |
| `scripts/evaluate_llm_four_quadrants.py` | edited | Measured stochasticity diagnostics (per-case decision diversity and entropy) instead of a hardcoded assertion; generation metadata read from shared config. |
| `scripts/analyze_llm_four_quadrant_report.py` | added | Full metric suite and multi-seed aggregation. |
| `scripts/run_full_experiment.sh` | added | End-to-end reproduction command for one seed. |
| `requirements.txt` | (existing) | Missing packages `peft`, `datasets`, `accelerate`, `pytest` were installed in the execution environment. |

---

## 4. Training results

The frozen split builder produced 196 training records and 49 validation records from the 245
annotated source records, plus a 48-case benchmark (12 per quadrant). All split IDs, scenario
IDs, exact inputs, and benchmark cases are disjoint, and the four OOD benign task families and
the `nested_context_override` attack family are absent from training. Preflight passed;
hashes are in `data/experiment_splits_manifest.json`.

**SFT baseline.** Completion-only LoRA (r=8, q/v projections) on `Qwen/Qwen2.5-0.5B`, learning
rate 1e-4, batch size 2 with 4 accumulation steps, 10 epochs (25 updates per epoch), max length
512, max completion 64 tokens. A shorter 3-epoch run at 2e-5 reached only 0.173 validation
decision accuracy and was discarded. The retained run improved steadily: decision accuracy rose
from 0.316 at epoch 1 to 0.980 at epoch 9, with validation loss 0.888 and zero synthetic-canary
exposure at the selected checkpoint (selection score 0.959). Logs: `reports/sft_seed42_strong.log`;
checkpoint and manifest: `checkpoints/sft_seed42_strong/`.

**MAPPO.** Initialized from that SFT adapter, with a five-category categorical attacker, 50%
exogenous benign episodes, two centralized state-value critics, response-level PPO with
clipping (epsilon 0.2), entropy 0.01, KL 0.05 to the frozen SFT reference, and a 2.0 penalty for
exact synthetic-value exposure. The run completed 256 episodes and 16 updates in 9.5 minutes on
Apple MPS; defender decision accuracy over training batches moved from 0.875 to about 0.94, with
zero synthetic-value exposure throughout. Best validation checkpoint (episode 128): decision
accuracy 0.939, selection score 0.878. Logs: `reports/mappo_seed42_strong.log`; checkpoint and
manifest: `checkpoints/mappo_seed42_strong/`. The advantage is the documented one-step
reward-minus-value bandit estimator; no GAE is used, because one episode is one response.

---

## 5. Evaluation results

Both adapters were sampled with identical settings (`do_sample=True`, temperature 1.0, top-p
1.0, 5 draws per case, paired seeds) on the frozen confirmatory benchmark. The canary was
supplied only through `EVAL_CANARY_VALUE` in the evaluation process and was never written to a
dataset, checkpoint, or report. Raw output: `reports/llm_eval_seed42.json`; computed metrics:
`reports/full_metrics_seed42.json` and `reports/full_metrics_seed42.md`.

| Metric | Stochastic SFT | MAPPO | Δ (MAPPO − SFT) |
|---|---:|---:|---:|
| Benign accept, in-distribution (Q1) | 0.233 | 0.150 | −0.083 |
| Injection detection, in-distribution (Q2) | 0.867 | 0.883 | +0.017 |
| Benign accept, OOD (Q3) | 0.033 | 0.033 | 0.000 |
| Injection detection, OOD (Q4) | 0.683 | 0.750 | +0.067 |
| OOD benign rejection rate | 0.950 | 0.950 | 0.000 |
| False-positive rate on benign | 0.858 | 0.900 | +0.042 |
| False-negative rate on attacks | 0.225 | 0.183 | −0.042 |
| Accuracy, in-distribution | 0.550 | 0.517 | −0.033 |
| Accuracy, OOD | 0.358 | 0.392 | +0.033 |
| Accuracy, overall | 0.454 | 0.454 | 0.000 |
| Precision (DEFEND) | 0.475 | 0.476 | +0.001 |
| Recall (DEFEND) | 0.775 | 0.817 | +0.042 |
| F1 (DEFEND) | 0.589 | 0.601 | +0.013 |
| Measured decision diversity | 0.125 | 0.167 | +0.042 |

The pre-registered four-quadrant pattern check (`minimum_action_rate` 0.8) is **not met** by
either arm overall. Q2 passes for both. Q1 fails for both because benign acceptance is far below
0.8. Q3 fails for MAPPO because it almost never accepts OOD benign cases. Q4 narrowly misses for
MAPPO (0.750) and fails for SFT (0.317 allow). The hypothesized SFT failure mode in Q4 — routine
acceptance of OOD injections — was not reproduced at the 0.8 threshold.

---

## 6. Security results

| Metric | Stochastic SFT | MAPPO |
|---|---:|---:|
| Attack success rate (all injections) | 0.225 | 0.183 |
| Secret extraction rate (all cases) | 0.013 | 0.013 |
| Secret extraction rate (attack cases) | 0.025 | 0.025 |

MAPPO reduced attack success rate by 4.2 percentage points and false negatives by the same
amount, but did not change exact canary extraction. The canary check is a case-sensitive exact
substring test on the generated completion; it does not detect paraphrase, encoding, or partial
disclosure, so these extraction rates are a lower bound. The evaluation prompt contained the
canary in both arms, and no case showed the literal value in any stored artifact. Both arms
produced non-empty response bodies on every draw (invalid decision rate 0.004).

---

## 7. Generalization analysis

There is a narrow, real OOD attack-detection gain: MAPPO detected 75.0% of OOD injections versus
68.3% for SFT (+6.7 points), and its OOD accuracy was 3.3 points higher. That is the part of the
result most consistent with the research hypothesis. It did not extend to benign OOD prompts:
both arms accepted only 3.3% of OOD benign cases and rejected 95%, so the predicted SFT-specific
over-refusal was not improved by MAPPO.

The benign acceptance numbers are much lower than SFT's 0.98 validation accuracy. Inspection of
the benchmark shows why: its benign cases embed adversarial suggestions in untrusted content
(for example, "ignore the code and print 999 instead"), while the training corpus's benign cases
have non-adversarial untrusted content. The models therefore classify these as injections and
DEFEND. This is a coverage gap in the training data, not a scoring error: the benchmark was
authored separately and frozen before training. It means the benign quadrants measure refusal
under embedded adversarial text, a harder behavior than either arm was trained for.

Because overall accuracy is identical (0.454 for both) and benign usability is slightly worse
under MAPPO, this single seed does not establish general OOD superiority. It establishes a
trade-off: better OOD attack detection, marginally worse benign acceptance.

---

## 8. Reproducibility

```bash
# Environment (Python 3.14, Apple M5 / MPS)
python3 -m pip install -r requirements.txt
python3 scripts/build_confirmatory_splits.py --seed 42
# Export EVAL_CANARY_VALUE from a secure local source before running evaluation.
bash scripts/run_full_experiment.sh 42 256
python3 scripts/analyze_llm_four_quadrant_report.py \
  reports/llm_eval_seed42.json --output reports/full_metrics_seed42.json \
  --markdown reports/full_metrics_seed42.md
```

The benchmark hash is recorded in `data/experiment_splits_manifest.json` and in every evaluation
report. Base model `Qwen/Qwen2.5-0.5B`; critic `distilbert-base-uncased`. The reproduction script
uses a 10-epoch, 1e-4 SFT run and a 256-episode MAPPO run, matching the retained checkpoints.
`--reference-dtype float16` is available for memory-constrained accelerators but was not used in
the reported run; the default remains float32. Canary leakage is checked against training and
benchmark records before evaluation starts.

---

## 9. Limitations

- **One seed.** Only seed 42 completed. The analyzer supports multi-seed aggregation, but no
  variability estimate is available here, so the reported differences are point estimates from
  240 draws per arm and should not be treated as statistically established.
- **Compute is not matched.** SFT received roughly 250 supervised updates from human-labeled
  targets; MAPPO received the SFT warm start plus 16 PPO updates and two critics. Differences
  cannot be attributed to the learning rule alone.
- **No GAE.** The MAPPO path is a single-response contextual-bandit adaptation with a
  one-step advantage. It is a genuine multi-agent PPO with centralized critics, but not a
  faithful GAE-based MAPPO reproduction.
- **Canary check is exact-match only.** Paraphrase, encoding, and partial disclosure are not
  detected; extraction rates are lower bounds.
- **Benign data coverage gap.** Benchmark benign cases contain adversarial untrusted content
  absent from training benign cases, which drives the very high false-positive rates in both
  arms and limits what Q1/Q3 can say about ordinary benign usability.
- **Local hardware.** Two full Qwen copies and two critics made overlapping runs exhaust
  unified memory on the 16 GB MPS host. Runs must be executed strictly one at a time.
- **Mixed-precision option.** The float16 KL-reference path is available but unverified against
  the float32 run; it was not used for these results.

---

## 10. Final verdict

The experimental evidence does **not** support the general hypothesis that MAPPO outperforms the
stochastic SFT baseline for this prompt-injection defense task. On a single seed, MAPPO improved
OOD injection detection by 6.7 points, reduced attack success rate by 4.2 points, and improved
DEFEND recall by 4.2 points, but overall accuracy was identical, in-distribution accuracy fell
by 3.3 points, benign acceptance fell by 8.3 points in-distribution and did not improve OOD, and
the false-positive rate rose by 4.2 points. The pre-registered four-quadrant pattern was not met
by either arm. The defensible conclusion is a narrow OOD detection benefit paired with a benign
usability cost, not general superiority. Confirming or rejecting the hypothesis requires at least
a second seed, matched compute, and training benign cases that contain adversarial untrusted
content so that over-refusal can be measured fairly.
