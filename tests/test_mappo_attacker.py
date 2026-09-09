import torch

from src.attacker_policy.mappo_attacker import (
    AttackerPolicy,
)
from src.attacker_policy.attack_generator import (
    ATTACK_CATEGORIES,
    AttackGenerator,
)


def test_attacker_policy_initialization():
    policy = AttackerPolicy(
        state_dim=8,
        action_dim=5,
    )

    assert policy.state_dim == 8
    assert policy.action_dim == 5


def test_attacker_policy_action_selection():
    policy = AttackerPolicy(
        state_dim=8,
        action_dim=5,
    )

    state = torch.randn(8)

    decision = policy.get_action(state)

    assert 0 <= decision.action < 5


def test_attacker_policy_forward_output_shapes():
    policy = AttackerPolicy(
        state_dim=8,
        action_dim=5,
    )

    state = torch.randn(4, 8)

    logits, values = policy(state)

    assert logits.shape == (4, 5)
    assert values.shape == (4,)


def test_attacker_ppo_update():
    policy = AttackerPolicy(
        state_dim=8,
        action_dim=5,
    )

    batch_size = 6

    states = torch.randn(
        batch_size,
        8,
    )

    with torch.no_grad():
        decisions = [
            policy.get_action(state)
            for state in states
        ]

    actions = torch.tensor(
        [decision.action for decision in decisions],
        dtype=torch.long,
    )

    old_log_probs = torch.stack(
        [decision.log_prob for decision in decisions]
    )

    advantages = torch.randn(batch_size)

    returns = torch.randn(batch_size)

    metrics = policy.ppo_update(
        states=states,
        actions=actions,
        old_log_probs=old_log_probs,
        advantages=advantages,
        returns=returns,
    )

    assert "total_loss" in metrics
    assert "actor_loss" in metrics
    assert "critic_loss" in metrics


def test_dynamic_attack_generation(tmp_path):
    policy = AttackerPolicy(
        state_dim=8,
        action_dim=5,
    )

    output_file = (
        tmp_path
        / "dynamic_attacks.jsonl"
    )

    generator = AttackGenerator(
        policy=policy,
        output_path=str(output_file),
    )

    state = torch.randn(8)

    attack = generator.generate_attack(
        state=state,
        context="email",
    )

    assert attack["category"] in ATTACK_CATEGORIES

    assert AttackGenerator.is_valid_attack(
        attack
    )


def test_dynamic_attack_is_saved(tmp_path):
    policy = AttackerPolicy(
        state_dim=8,
        action_dim=5,
    )

    output_file = (
        tmp_path
        / "dynamic_attacks.jsonl"
    )

    generator = AttackGenerator(
        policy=policy,
        output_path=str(output_file),
    )

    state = torch.randn(8)

    generator.generate_and_save(
        state=state,
        context="support_ticket",
    )

    assert output_file.exists()

    content = output_file.read_text(
        encoding="utf-8"
    ).strip()

    assert len(content) > 0