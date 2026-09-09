from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
from torch.distributions import Categorical


@dataclass
class AttackDecision:
    """Represents one action selected by the attacker policy."""

    action: int
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor


class AttackerPolicy(nn.Module):
    """
    Actor-Critic network for selecting an attack category.

    This is the Member 1 attacker-side policy used in Review 2.
    The actor selects an attack action, while the critic estimates
    the value of the current state.
    """

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 5,
        hidden_dim: int = 64,
        learning_rate: float = 3e-4,
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.feature_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
        )

    def forward(self, state: torch.Tensor):
        """
        Returns:
            logits: Attack action logits.
            value: Estimated value of the current state.
        """

        if state.dim() == 1:
            state = state.unsqueeze(0)

        features = self.feature_network(state.float())

        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)

        return logits, value

    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> AttackDecision:
        """Select an attack action from the current policy."""

        logits, value = self.forward(state)

        distribution = Categorical(logits=logits)

        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = distribution.sample()

        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()

        return AttackDecision(
            action=int(action.squeeze().item()),
            log_prob=log_prob.squeeze(),
            entropy=entropy.squeeze(),
            value=value.squeeze(),
        )

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Evaluates actions for PPO/MAPPO-style updates.
        """

        logits, values = self.forward(states)

        distribution = Categorical(logits=logits)

        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()

        return log_probs, values, entropy

    def ppo_update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        clip_epsilon: float = 0.2,
        value_coefficient: float = 0.5,
        entropy_coefficient: float = 0.01,
    ) -> Dict[str, float]:
        """
        Performs one PPO-style update of the attacker policy.
        """

        log_probs, values, entropy = self.evaluate_actions(
            states,
            actions,
        )

        ratio = torch.exp(log_probs - old_log_probs)

        surrogate_1 = ratio * advantages

        surrogate_2 = torch.clamp(
            ratio,
            1.0 - clip_epsilon,
            1.0 + clip_epsilon,
        ) * advantages

        actor_loss = -torch.min(
            surrogate_1,
            surrogate_2,
        ).mean()

        critic_loss = nn.functional.mse_loss(
            values,
            returns,
        )

        entropy_bonus = entropy.mean()

        total_loss = (
            actor_loss
            + value_coefficient * critic_loss
            - entropy_coefficient * entropy_bonus
        )

        self.optimizer.zero_grad()

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.parameters(),
            max_norm=1.0,
        )

        self.optimizer.step()

        return {
            "total_loss": float(total_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "entropy": float(entropy_bonus.item()),
        }