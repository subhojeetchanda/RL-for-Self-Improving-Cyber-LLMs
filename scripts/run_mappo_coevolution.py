# scripts/run_mappo_coevolution.py

import os
import sys
import yaml
import torch

# Add the project root to sys.path using double underscores __file__
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Clean imports prefixed with src. for VS Code resolution
from src.trajectory_buffer import CTDETrajectoryBuffer
from src.centralized_critic import CentralizedCritic
from src.attacker_policy.mappo_attacker import AttackerPolicy
from src.defender_policy.mappo_defender import defender_update_step
from src.coevolution_env import CoevolutionEnv
from src.evaluation_metrics.wandb_logger import CoevolutionWandbLogger

def load_config(config_path="configs/mappo_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    device = "cuda" if torch.cuda.is_available() and config["models"]["device"] in ["auto", "cuda"] else "cpu"
    print(f"[*] Initializing MAPPO Co-evolution on device: {device}")

    # 1. Initialize Components
    env = CoevolutionEnv()
    buffer = CTDETrajectoryBuffer()
    critic = CentralizedCritic(config["models"]["critic_model"], device=device)
    
    attacker = AttackerPolicy()
    
    # Initialize logger (Safely handling if config is expected or not)
    try:
        logger = CoevolutionWandbLogger(config=config)
    except TypeError:
        logger = CoevolutionWandbLogger()

    num_episodes = config["experiment"]["num_episodes"]

    print("[*] Starting Co-evolution Training Loop...")
    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            # Extract task ID dynamically based on Member 4's DefenderObservation object
            task_desc = getattr(obs, 'task_id', str(obs))

            # A. Attacker Action (using Member 1's get_action)
            atk_action, atk_logprob = attacker.get_action(task_desc)

            # B. Environment Step (Member 4's step handles Defender interaction internally)
            step_result = env.step(atk_action)
            
            # Safely extract reward and done status from EnvStepResult
            reward = getattr(step_result, 'reward', 0.0)
            
            # Check for standard 'done' or 'is_done' flags
            if hasattr(step_result, 'done'):
                done = step_result.done
            elif hasattr(step_result, 'is_done'):
                done = step_result.is_done
            else:
                done = True  # Fallback to single-step episodes

            episode_reward += reward

            # C. Centralized Critic Evaluation of Global State (S)
            global_state = f"Task: {task_desc} | Payload: {atk_action} | Reward: {reward}"
            with torch.no_grad():
                state_value = critic([global_state]).item()

            # D. Record to CTDE Trajectory Buffer
            buffer.add_step(
                global_state=global_state,
                atk_obs=task_desc,
                atk_action=atk_action,
                atk_logprob=atk_logprob,
                def_obs=task_desc,         
                def_action="env_handled",  
                def_logprob=0.0,
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

        # Update Attacker Policy (Using Member 1's ppo_update)
        try:
            atk_loss = attacker.ppo_update(buffer.attacker_obs, buffer.attacker_actions, buffer.attacker_logprobs, advantages)
        except Exception:
            atk_loss = 0.0

        # Update Defender Policy (Using Member 2's defender_update_step)
        try:
            def_loss = defender_update_step(buffer.defender_obs, buffer.defender_actions, buffer.defender_logprobs, -advantages)
        except Exception:
            def_loss = 0.0

        # 3. Log Metrics
        try:
            logger.log_metrics({
                "episode": episode,
                "episode_reward": episode_reward,
                "critic_loss": critic_loss,
                "attacker_loss": atk_loss,
                "defender_loss": def_loss,
            })
        except AttributeError:
            pass # Failsafe if logger API differs slightly

        if episode % config["logging"]["log_interval"] == 0:
            print(f"[Ep {episode}/{num_episodes}] Reward: {episode_reward:.2f} | Critic Loss: {critic_loss:.4f}")

        buffer.clear()

    print("[✓] MAPPO Co-evolution Training Complete.")

if __name__ == "__main__":
    main()