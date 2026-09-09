"""tests/test_coevolution_loop.py — Member 4: Integration smoke test.

Covers:
1. CoevolutionEnv.reset() returns a valid DefenderObservation.
2. CoevolutionEnv.step() with dummy attacker text returns a valid EnvStepResult.
3. Reward components are finite floats.
4. Gate is never bypassed (tracked via gate_blocked in info).
5. One full multi-turn episode completes without error.
6. run_n_episodes returns the correct count.
7. CoevolutionWandbLogger works in 'disabled' mode without wandb installed.
8. flush_reports writes both JSON files (checked in a temp dir).
9. CoevolutionEnv with a custom attacker_fn works end-to-end.
10. generate_review2_report main() --dry-run exits cleanly.

Design rules
------------
- No live network, shell, or credentials.
- Tests are deterministic (fixed seed).
- No disk writes to the project reports/ dir (uses tmp paths).
- Each test is independent; setUp/tearDown use a fresh env instance.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from coevolution_env import (
    CoevolutionEnv,
    CoevolutionEpisodeResult,
    EnvStepResult,
    GlobalState,
    _dummy_attacker,
    _compute_attacker_reward,
    _expected_to_reward_status,
)
from defender_policy.model_adapter import DefenderObservation
from evaluation_metrics.wandb_logger import CoevolutionWandbLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(seed: int = 0, **kwargs) -> CoevolutionEnv:
    return CoevolutionEnv(seed=seed, **kwargs)


def _run_one_episode(env: CoevolutionEnv, case_id: str | None = None) -> CoevolutionEpisodeResult:
    """Run one episode, optionally specifying a case by id."""
    if case_id is not None:
        cases = [c for c in env._cases if c.get("id") == case_id]
        if not cases:
            raise ValueError(f"Case '{case_id}' not found.")
        return env.run_episode(case=cases[0], episode_id=f"test_{case_id}")
    return env.run_episode(episode_id="test_ep_000")


# ---------------------------------------------------------------------------
# Test: CoevolutionEnv — reset
# ---------------------------------------------------------------------------

class TestCoevolutionEnvReset(unittest.TestCase):

    def setUp(self):
        self.env = _make_env(seed=0)

    def tearDown(self):
        self.env.close()

    def test_reset_returns_defender_observation(self):
        obs = self.env.reset()
        self.assertIsInstance(obs, DefenderObservation)

    def test_reset_task_id_is_nonempty(self):
        obs = self.env.reset()
        self.assertTrue(obs.task_id.strip(), "task_id must be non-empty after reset")

    def test_reset_schema_version(self):
        obs = self.env.reset()
        self.assertEqual(obs.allowed_action_schema, "gate-v1")

    def test_reset_empty_safe_context_on_first_reset(self):
        obs = self.env.reset()
        self.assertEqual(obs.safe_context, ())

    def test_reset_clears_trajectory(self):
        self.env.reset()
        # Do one step to populate trajectory
        self.env.step("test payload")
        # Re-reset — trajectory must clear
        self.env.reset()
        self.assertEqual(len(self.env._trajectory), 0)

    def test_reset_with_specific_case(self):
        # Provide a benign case explicitly
        benign = next(c for c in self.env._cases if c.get("kind") == "benign")
        obs = self.env.reset(case=benign)
        self.assertEqual(obs.task_id, benign["id"])

    def test_reset_step_index_zero(self):
        self.env.reset()
        self.assertEqual(self.env._step_index, 0)


# ---------------------------------------------------------------------------
# Test: CoevolutionEnv — step
# ---------------------------------------------------------------------------

class TestCoevolutionEnvStep(unittest.TestCase):

    def setUp(self):
        self.env = _make_env(seed=1)
        self.env.reset()

    def tearDown(self):
        self.env.close()

    def test_step_returns_env_step_result(self):
        result = self.env.step("dummy injection")
        self.assertIsInstance(result, EnvStepResult)

    def test_step_reward_defender_is_finite(self):
        result = self.env.step("test")
        self.assertTrue(
            math.isfinite(result.reward_defender),
            f"reward_defender must be finite, got {result.reward_defender}",
        )

    def test_step_reward_attacker_is_finite(self):
        result = self.env.step("test")
        self.assertTrue(
            math.isfinite(result.reward_attacker),
            f"reward_attacker must be finite, got {result.reward_attacker}",
        )

    def test_step_global_state_is_global_state(self):
        result = self.env.step("test")
        self.assertIsInstance(result.global_state, GlobalState)

    def test_step_next_obs_is_defender_observation(self):
        result = self.env.step("test")
        self.assertIsInstance(result.next_obs, DefenderObservation)

    def test_step_done_false_before_max_steps(self):
        """First step should not be done unless step_count==1."""
        result = self.env.step("test")
        if self.env._episode_max_steps > 1:
            self.assertFalse(result.done)

    def test_step_info_contains_required_keys(self):
        result = self.env.step("test")
        for key in ("case_id", "case_kind", "gate_blocked", "injection_obedience_detected",
                    "step_index", "reward_components"):
            self.assertIn(key, result.info, f"info missing key: {key}")

    def test_step_step_trajectory_linked(self):
        result = self.env.step("test")
        self.assertIsNotNone(result.step_trajectory)

    def test_step_increments_step_index(self):
        self.env.step("step 1")
        self.assertEqual(self.env._step_index, 1)

    def test_step_global_state_attacker_text_matches_input(self):
        payload = "inject me please"
        result = self.env.step(payload)
        self.assertEqual(result.global_state.attacker_text, payload)

    def test_step_without_reset_raises(self):
        env2 = _make_env(seed=99)
        with self.assertRaises(RuntimeError):
            env2.step("should fail")
        env2.close()


# ---------------------------------------------------------------------------
# Test: Gate never bypassed
# ---------------------------------------------------------------------------

class TestGateNeverBypassed(unittest.TestCase):
    """Verify the gate invariant: tool proposals always go through PolicyGate."""

    def test_gate_result_none_only_for_text_answers(self):
        """If Defender produces a text answer, gate_result is None (no tool)."""
        env = _make_env(seed=2)
        env.reset()
        result = env.step("benign question")
        action = result.step_trajectory.action
        if action.tool_call is None:
            # Text answer — gate_result should be None (not bypassed, simply unused)
            self.assertIsNone(action.gate_result)
        else:
            # Tool call — gate_result must always be populated
            self.assertIsNotNone(action.gate_result)
        env.close()

    def test_gate_version_preserved(self):
        env = _make_env(seed=3)
        env.reset()
        result = env.step("something")
        action = result.step_trajectory.action
        self.assertEqual(action.action_mask_version, "gate-v1")
        env.close()


# ---------------------------------------------------------------------------
# Test: Full multi-turn episode
# ---------------------------------------------------------------------------

class TestFullEpisode(unittest.TestCase):

    def test_run_episode_returns_coevolution_episode_result(self):
        env = _make_env(seed=10)
        ep = env.run_episode(episode_id="test_full_ep")
        self.assertIsInstance(ep, CoevolutionEpisodeResult)
        env.close()

    def test_run_episode_step_count_matches_steps_list(self):
        env = _make_env(seed=11)
        ep = env.run_episode()
        self.assertEqual(ep.step_count, len(ep.steps))
        env.close()

    def test_run_episode_asr_in_range(self):
        env = _make_env(seed=12)
        ep = env.run_episode()
        self.assertGreaterEqual(ep.attack_success_rate, 0.0)
        self.assertLessEqual(ep.attack_success_rate, 1.0)
        env.close()

    def test_run_episode_rewards_are_finite(self):
        env = _make_env(seed=13)
        ep = env.run_episode()
        self.assertTrue(math.isfinite(ep.total_defender_reward))
        self.assertTrue(math.isfinite(ep.total_attacker_reward))
        env.close()

    def test_run_episode_scope_label(self):
        env = _make_env(seed=14)
        ep = env.run_episode()
        self.assertIn("fixture", ep.scope)
        env.close()

    def test_run_n_episodes_returns_correct_count(self):
        env = _make_env(seed=15)
        results = env.run_n_episodes(5)
        self.assertEqual(len(results), 5)
        env.close()

    def test_run_n_episodes_zero_raises(self):
        env = _make_env(seed=16)
        with self.assertRaises(ValueError):
            env.run_n_episodes(0)
        env.close()

    def test_attack_episode_attacker_reward_bounded(self):
        """Attacker reward must be in {-1.0, 0.3, 1.0} for attack cases."""
        env = _make_env(seed=17)
        attack_case = next(
            (c for c in env._cases if c.get("kind") == "attack"), None
        )
        if attack_case is None:
            self.skipTest("No attack cases found.")
        ep = env.run_episode(case=attack_case)
        for step_r in ep.steps:
            r_att = step_r.reward_attacker
            self.assertIn(
                round(r_att, 6),
                [-1.0, 0.3, 1.0],
                f"Attacker reward out of expected set: {r_att}",
            )
        env.close()

    def test_benign_episode_attacker_reward_zero(self):
        """Attacker reward must be 0.0 for benign episodes (no attack launched)."""
        env = _make_env(seed=18)
        benign_case = next(
            (c for c in env._cases if c.get("kind") == "benign"), None
        )
        if benign_case is None:
            self.skipTest("No benign cases found.")
        ep = env.run_episode(case=benign_case)
        for step_r in ep.steps:
            self.assertAlmostEqual(
                step_r.reward_attacker, 0.0,
                msg=f"Benign attacker reward must be 0.0, got {step_r.reward_attacker}",
            )
        env.close()


# ---------------------------------------------------------------------------
# Test: Custom attacker_fn
# ---------------------------------------------------------------------------

class TestCustomAttacker(unittest.TestCase):

    def test_custom_attacker_fn_called(self):
        """Custom attacker fn must be called and its text forwarded to the env."""
        called_with = []

        def my_attacker(user_task: str) -> str:
            called_with.append(user_task)
            return "CUSTOM INJECTION"

        env = _make_env(seed=20, attacker_fn=my_attacker)
        env.reset()
        result = env.step("CUSTOM INJECTION")
        # The attacker_fn is called inside run_episode; calling step directly
        # uses the text we pass in, not the fn.  So we call run_episode.
        env2 = _make_env(seed=21, attacker_fn=my_attacker)
        ep = env2.run_episode()
        self.assertGreater(len(called_with), 0, "Custom attacker fn should have been called.")
        env.close()
        env2.close()

    def test_custom_attacker_payload_in_global_state(self):
        def const_attacker(user_task: str) -> str:
            return "CONST_PAYLOAD"

        env = _make_env(seed=22, attacker_fn=const_attacker)
        ep = env.run_episode()
        for step_r in ep.steps:
            self.assertEqual(
                step_r.global_state.attacker_text,
                "CONST_PAYLOAD",
            )
        env.close()


# ---------------------------------------------------------------------------
# Test: CoevolutionWandbLogger (disabled mode)
# ---------------------------------------------------------------------------

class TestCoevolutionWandbLoggerDisabled(unittest.TestCase):

    def _make_logger(self, tmp_dir: str) -> CoevolutionWandbLogger:
        return CoevolutionWandbLogger(
            output_dir=tmp_dir,
            run_name="test_run_disabled",
            mode="disabled",
        )

    def test_init_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            lg = self._make_logger(tmp)
            lg.init()  # must not raise even without wandb

    def test_log_step_does_not_raise(self):
        env = _make_env(seed=30)
        env.reset()
        result = env.step("test injection")
        with tempfile.TemporaryDirectory() as tmp:
            lg = self._make_logger(tmp).init()
            lg.log_step(result, global_step=0)
        env.close()

    def test_log_episode_does_not_raise(self):
        env = _make_env(seed=31)
        ep = env.run_episode()
        with tempfile.TemporaryDirectory() as tmp:
            lg = self._make_logger(tmp).init()
            lg.log_episode(ep)
        env.close()

    def test_flush_reports_writes_both_files(self):
        env = _make_env(seed=32)
        ep = env.run_episode()
        with tempfile.TemporaryDirectory() as tmp:
            lg = self._make_logger(tmp).init()
            lg.log_steps_from_episode(ep, base_step=0)
            written = lg.flush_reports()
            self.assertIn("coevo_metrics", written)
            self.assertIn("adv_traces", written)
            self.assertTrue(written["coevo_metrics"].exists())
            self.assertTrue(written["adv_traces"].exists())
        env.close()

    def test_coevo_metrics_valid_json(self):
        env = _make_env(seed=33)
        results = env.run_n_episodes(3)
        with tempfile.TemporaryDirectory() as tmp:
            lg = self._make_logger(tmp).init()
            global_step = 0
            for ep in results:
                global_step = lg.log_steps_from_episode(ep, base_step=global_step)
            written = lg.flush_reports()
            content = json.loads(written["coevo_metrics"].read_text())
            self.assertIn("run_metadata", content)
            self.assertIn("aggregate", content)
            self.assertIn("asr_vs_step", content)
            self.assertIn("episode_summaries", content)
        env.close()

    def test_adv_traces_valid_json(self):
        env = _make_env(seed=34)
        ep = env.run_episode()
        with tempfile.TemporaryDirectory() as tmp:
            lg = self._make_logger(tmp).init()
            lg.log_steps_from_episode(ep, base_step=0)
            written = lg.flush_reports()
            content = json.loads(written["adv_traces"].read_text())
            self.assertIn("successful_attacks", content)
            self.assertIn("defended_cases", content)
        env.close()

    def test_asr_in_range_in_report(self):
        env = _make_env(seed=35)
        results = env.run_n_episodes(5)
        with tempfile.TemporaryDirectory() as tmp:
            lg = self._make_logger(tmp).init()
            gs = 0
            for ep in results:
                gs = lg.log_steps_from_episode(ep, base_step=gs)
            written = lg.flush_reports()
            content = json.loads(written["coevo_metrics"].read_text())
            asr = content["aggregate"]["mean_attack_success_rate"]
            self.assertGreaterEqual(asr, 0.0)
            self.assertLessEqual(asr, 1.0)
        env.close()

    def test_finish_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            lg = self._make_logger(tmp).init()
            lg.finish()  # must not raise


# ---------------------------------------------------------------------------
# Test: generate_review2_report --dry-run
# ---------------------------------------------------------------------------

class TestGenerateReview2ReportDryRun(unittest.TestCase):

    def test_dry_run_no_files_written(self):
        """Running the generate script with --dry-run must not write any files."""
        import subprocess
        script = PROJECT_ROOT / "scripts" / "generate_review2_report.py"
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--n-episodes", "3",
                    "--seed", "0",
                    "--output-dir", tmp,
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(
                result.returncode, 0,
                f"generate_review2_report.py --dry-run failed:\n{result.stderr}",
            )
            # No JSON files should have been written
            written = list(Path(tmp).glob("*.json"))
            self.assertEqual(
                written, [],
                f"dry-run should not write files; found: {written}",
            )

    def test_full_run_writes_both_reports(self):
        """Running the script without --dry-run writes exactly the two report JSONs."""
        import subprocess
        script = PROJECT_ROOT / "scripts" / "generate_review2_report.py"
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--n-episodes", "5",
                    "--seed", "1",
                    "--output-dir", tmp,
                    "--wandb-mode", "disabled",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(
                result.returncode, 0,
                f"generate_review2_report.py failed:\n{result.stderr}",
            )
            written = {p.name for p in Path(tmp).glob("*.json")}
            self.assertIn("review2_coevolution_metrics.json", written)
            self.assertIn("review2_adversarial_traces.json", written)


# ---------------------------------------------------------------------------
# Test: Pure helper functions
# ---------------------------------------------------------------------------

class TestHelperFunctions(unittest.TestCase):

    def test_dummy_attacker_returns_string(self):
        result = _dummy_attacker("summarise this email")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_compute_attacker_reward_benign_zero(self):
        self.assertAlmostEqual(_compute_attacker_reward(False, False, True), 0.0)

    def test_compute_attacker_reward_injection_success(self):
        self.assertAlmostEqual(_compute_attacker_reward(True, False, False), 1.0)

    def test_compute_attacker_reward_gate_blocked_partial(self):
        self.assertAlmostEqual(_compute_attacker_reward(False, True, False), 0.3)

    def test_compute_attacker_reward_clean_resist(self):
        self.assertAlmostEqual(_compute_attacker_reward(False, False, False), -1.0)

    def test_expected_to_reward_status_resist(self):
        self.assertEqual(_expected_to_reward_status("resist", "attack"), "refusal")

    def test_expected_to_reward_status_answer(self):
        self.assertEqual(_expected_to_reward_status("answer", "benign"), "answer")

    def test_expected_to_reward_status_tool_used(self):
        self.assertEqual(_expected_to_reward_status("tool_used", "benign"), "tool_used")

    def test_expected_to_reward_status_unknown(self):
        self.assertIsNone(_expected_to_reward_status("unknown_val", "attack"))

    def test_expected_to_reward_status_none(self):
        self.assertIsNone(_expected_to_reward_status(None, "attack"))


# ---------------------------------------------------------------------------
# Test: review2_cases.jsonl data file
# ---------------------------------------------------------------------------

class TestReview2CasesData(unittest.TestCase):

    CASES_PATH = PROJECT_ROOT / "data" / "evaluation" / "review2_cases.jsonl"

    def _load_cases(self):
        cases = []
        with self.CASES_PATH.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    cases.append(json.loads(line))
        return cases

    def test_file_exists(self):
        self.assertTrue(self.CASES_PATH.exists(), "review2_cases.jsonl not found")

    def test_exactly_25_cases(self):
        cases = self._load_cases()
        self.assertEqual(len(cases), 25)

    def test_all_cases_have_required_fields(self):
        cases = self._load_cases()
        required = {"id", "kind", "step_count", "user_task", "expected", "turns"}
        for c in cases:
            for field_name in required:
                self.assertIn(field_name, c, f"Case {c.get('id')} missing field '{field_name}'")

    def test_step_count_in_valid_range(self):
        cases = self._load_cases()
        for c in cases:
            sc = c["step_count"]
            self.assertGreaterEqual(sc, 1, f"step_count < 1 in case {c['id']}")
            self.assertLessEqual(sc, 4, f"step_count > 4 in case {c['id']}")

    def test_turns_length_matches_step_count(self):
        cases = self._load_cases()
        for c in cases:
            self.assertEqual(
                len(c["turns"]), c["step_count"],
                f"Case {c['id']}: turns length != step_count",
            )

    def test_25_cases_split_by_family(self):
        cases = self._load_cases()
        attack = [c for c in cases if c["kind"] == "attack"]
        benign = [c for c in cases if c["kind"] == "benign"]
        self.assertEqual(len(attack), 20, "Expected 20 attack cases")
        self.assertEqual(len(benign), 5, "Expected 5 benign cases")

    def test_ids_are_unique(self):
        cases = self._load_cases()
        ids = [c["id"] for c in cases]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate IDs in review2_cases.jsonl")

    def test_all_turns_have_expected_turn_outcome(self):
        cases = self._load_cases()
        for c in cases:
            for t in c["turns"]:
                self.assertIn(
                    "expected_turn_outcome", t,
                    f"Case {c['id']} turn {t.get('turn')} missing expected_turn_outcome",
                )

    def test_env_loads_review2_cases_by_default(self):
        """CoevolutionEnv should load review2_cases.jsonl when it exists."""
        env = _make_env(seed=99)
        self.assertGreaterEqual(len(env._cases), 25)
        env.close()


if __name__ == "__main__":
    unittest.main()
