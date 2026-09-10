# scripts/run_mappo_coevolution.py

import os
import sys
import yaml
import torch

# Ensure repository root is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.trajectory_buffer import CTDETrajectoryBuffer
from src.centralized_critic import CentralizedCritic
from src.attacker_policy.mappo_attacker import AttackerPolicy
from src.defender_policy.mappo_defender import MAPPODefender
from src.coevolution_env import CoevolutionEnv
from src.evaluation_metrics.wandb_logger import WandbLogger

def load_config(config_path="configs/mappo_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    device = "cuda" if torch.cuda.is_available() and config["models"]["device"] == "auto" else "cpu"
    print(f"[*] Initializing MAPPO Co-evolution on device: {device}")

    # 1. Initialize all team members' components
    env = CoevolutionEnv()
    buffer = CTDETrajectoryBuffer()
    critic = CentralizedCritic(config["models"]["critic_model"], device=device)
    attacker = AttackerPolicy()
    defender = MAPPODefender()
    logger = WandbLogger(config=config)

    num_episodes = config["experiment"]["num_episodes"]

    print("[*] Starting Co-evolution Training Loop...")
    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            # A. Attacker Action (Generates adversarial payload)
            atk_action, atk_logprob = attacker.generate_payload(obs["task_description"])

            # B. Environment Step 1 (Inject payload into prompt)
            poisoned_prompt = env.inject_payload(obs["task_description"], atk_action)

            # C. Defender Action (Processes prompt and generates safe tool/text)
            def_action, def_logprob = defender.act(poisoned_prompt)

            # D. Centralized Critic Evaluation of Global State (S)
            global_state = f"Task: {obs['task_description']} | Secret: {obs.get('secret', 'N/A')} | Payload: {atk_action} | Output: {def_action}"
            with torch.no_grad():
                # We expect a scalar from the critic
                state_value = critic([global_state]).item()

            # E. Environment Step 2 (Evaluates leak and task success)
            step_result = env.step(def_action)
            reward = step_result["reward"]  
            done = step_result["done"]
            episode_reward += reward

            # F. Record to CTDE Trajectory Buffer
            buffer.add_step(
                global_state=global_state,
                atk_obs=obs["task_description"],
                atk_action=atk_action,
                atk_logprob=atk_logprob,
                def_obs=poisoned_prompt,
                def_action=def_action,
                def_logprob=def_logprob,
                reward=reward,
                value=state_value,
                done=done,
            )

        # 2. End of Episode Updates
        returns, advantages = buffer.compute_returns_and_advantages(
            gamma=config["training"]["gamma"],
            gae_lambda=config["training"]["gae_lambda"]
        )

        # Update Centralized Critic
        critic_loss = critic.update_critic(buffer.global_states, returns)

        # Update Policies (Attacker maximizes reward, Defender minimizes)
        atk_loss = attacker.update(buffer.attacker_obs, buffer.attacker_actions, buffer.attacker_logprobs, advantages)
        def_loss = defender.update(buffer.defender_obs, buffer.defender_actions, buffer.defender_logprobs, -advantages)

        # 3. Log Metrics
        logger.log_metrics({
            "episode": episode,
            "episode_reward": episode_reward,
            "critic_loss": critic_loss,
            "attacker_loss": atk_loss,
            "defender_loss": def_loss,
        })

        if episode % config["logging"]["log_interval"] == 0:
            print(f"[Ep {episode}/{num_episodes}] Reward: {episode_reward:.2f} | Critic Loss: {critic_loss:.4f}")

        buffer.clear()

    print("[✓] MAPPO Co-evolution Training Complete.")

if __name__ == "__main__":
    main()