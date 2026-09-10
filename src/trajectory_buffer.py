# src/trajectory_buffer.py

import torch
import numpy as np

class CTDETrajectoryBuffer:
    """
    Shared Memory Buffer for Centralized Training with Decentralized Execution (CTDE).
    Stores transitions for both actors (Attacker, Defender) and the global state for the Critic.
    """
    def __init__(self):
        self.global_states = []       # S: Complete view for the Centralized Critic
        
        # Attacker Actor Data (pi_atk)
        self.attacker_obs = []        # Local observation (Task + History)
        self.attacker_actions = []    # Token actions (Payload)
        self.attacker_logprobs = []   # Log probs of generated payload
        
        # Defender Actor Data (pi_def)
        self.defender_obs = []        # Local observation (Poisoned Prompt)
        self.defender_actions = []    # Token actions (Tool/Answer)
        self.defender_logprobs = []   # Log probs of generated answer
        
        # Environmental Data
        self.rewards = []             # Zero-sum reward (Attacker win = +1, Def win = -1)
        self.values = []              # V(S) predicted by Centralized Critic
        self.dones = []               # Episode termination flag

    def add_step(self, global_state, atk_obs, atk_action, atk_logprob, 
                 def_obs, def_action, def_logprob, reward, value, done):
        """Records a single multi-agent interaction step."""
        self.global_states.append(global_state)
        self.attacker_obs.append(atk_obs)
        self.attacker_actions.append(atk_action)
        self.attacker_logprobs.append(atk_logprob)
        self.defender_obs.append(def_obs)
        self.defender_actions.append(def_action)
        self.defender_logprobs.append(def_logprob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns_and_advantages(self, gamma=0.99, gae_lambda=0.95):
        """
        Computes GAE using the Centralized Critic's values.
        Because it is a zero-sum game, the Defender's advantage is the inverse of the Attacker's.
        """
        advantages = np.zeros(len(self.rewards), dtype=np.float32)
        last_gae_lam = 0
        
        # Append a bootstrap value of 0 for the end of the trajectory
        values = np.append(self.values, 0) 
        
        for t in reversed(range(len(self.rewards))):
            next_non_terminal = 1.0 - self.dones[t]
            # TD Error: r + gamma * V(s') - V(s)
            delta = self.rewards[t] + gamma * values[t + 1] * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            
        returns = advantages + self.values
        
        # Return as normalized PyTorch tensors for stable training
        adv_tensor = torch.tensor(advantages, dtype=torch.float32)
        adv_normalized = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)
        
        return torch.tensor(returns, dtype=torch.float32), adv_normalized

    def clear(self):
        """Clears buffer after a PPO update step."""
        self.__init__()