# import numpy as np
# from stable_baselines3 import DDPG
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.ddpg.policies import MultiInputPolicy

# from imitation.algorithms import bc
# from imitation.data import rollout
# from imitation.data.wrappers import RolloutInfoWrapper
# from imitation.policies.serialize import load_policy
# from imitation.util.util import make_vec_env
# import gymnasium as gym
# import gymnasium_robotics
# rng = np.random.default_rng(0)

# # ⚠️ Set your environment
# gym.register_envs(gymnasium_robotics)
# env = make_vec_env("FetchReachDense-v4", n_envs=1, rng=rng)

# def train_expert():
#     print("Training an expert.")
#     expert = DDPG(
#         policy=MultiInputPolicy,  # use MultiInputPolicy for dict obs
#         env=env,
#         seed=0,
#         batch_size=64,
#         learning_rate=0.0003,
#         train_freq=(1, "episode"),
#         learning_starts=100,
#         verbose=1,
#     )
#     expert.learn(10000)  # or more for decent policy
#     return expert

# def sample_expert_transitions():
#     expert = train_expert()  # or use load_policy(...) for pre-trained
#     print("Sampling expert transitions.")
#     rollouts = rollout.rollout(
#         expert,
#         env,
#         rollout.make_sample_until(min_timesteps=None, min_episodes=50),
#         rng=rng,
#     )
#     return rollout.flatten_trajectories(rollouts)

# # --- Collect transitions ---
# transitions = sample_expert_transitions()
# print("Sampled transitions:", transitions[:1])

# # --- Initialize BC trainer ---
# bc_trainer = bc.BC(
#     observation_space=env.observation_space,
#     action_space=env.action_space,
#     demonstrations=transitions,
#     rng=rng,
# )

# # --- Evaluation environment ---
# evaluation_env = make_vec_env(
#     ENV_ID,
#     rng=rng,
#     env_make_kwargs={"render_mode": "human"},
# )

# # --- Evaluate before training ---
# print("Evaluating the untrained policy.")
# reward, _ = evaluate_policy(
#     bc_trainer.policy,
#     evaluation_env,
#     n_eval_episodes=3,
#     render=True,
# )
# print(f"Reward before training: {reward}")

# # --- Train ---
# print("Training with Behavior Cloning")
# bc_trainer.train(n_epochs=1)

# # --- Evaluate after training ---
# print("Evaluating the trained policy.")
# reward, _ = evaluate_policy(
#     bc_trainer.policy,
#     evaluation_env,
#     n_eval_episodes=3,
#     render=True,
# )
# print(f"Reward after training: {reward}")




"""This is a simple example demonstrating how to clone the behavior of an expert.

Refer to the jupyter notebooks for more detailed examples of how to use the algorithms.
"""
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env

rng = np.random.default_rng(0)
env = make_vec_env(
    "seals:seals/CartPole-v0",
    rng=rng,
    post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # for computing rollouts
)


def train_expert():
    # note: use `download_expert` instead to download a pretrained, competent expert
    print("Training a expert.")
    expert = PPO(
        policy=MlpPolicy,
        env=env,
        seed=0,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0003,
        n_epochs=10,
        n_steps=64,
    )
    expert.learn(1_000)  # Note: change this to 100_000 to train a decent expert.
    return expert


def download_expert():
    print("Downloading a pretrained expert.")
    expert = load_policy(
        "ppo-huggingface",
        organization="HumanCompatibleAI",
        env_name="seals-CartPole-v0",
        venv=env,
    )
    return expert


def sample_expert_transitions():
    # expert = train_expert()  # uncomment to train your own expert
    expert = download_expert()

    print("Sampling expert transitions.")
    rollouts = rollout.rollout(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=50),
        rng=rng,
    )
    print("ROLLOUTS: " + str(rollouts))
    return rollout.flatten_trajectories(rollouts)


transitions = sample_expert_transitions()

# print("TRANSITIONS ARE: " + str(transitions))
# for key in transitions._fields if hasattr(transitions, "_fields") else transitions.keys():
#     val = getattr(transitions, key) if hasattr(transitions, key) else transitions[key]
#     if torch.is_tensor(val):
#         print(f"{key}: Tensor on device {val.device}")
#     else:
#         print(f"{key}: type {type(val)} (likely numpy array, on CPU)")




bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    rng=rng,
)
print("OBSERVATION SPACE: " + str(env.observation_space))
print("ACTION SPACE: " + str(env.action_space))
print("DEMONSTRATIONS: " + str(transitions))


devices = {param.device for param in bc_trainer.policy.parameters()}
evaluation_env = make_vec_env(
    "seals:seals/CartPole-v0",
    rng=rng,
    env_make_kwargs={"render_mode": "human"},  # for rendering
)

print("Evaluating the untrained policy.")
reward, _ = evaluate_policy(
    bc_trainer.policy,  # type: ignore[arg-type]
    evaluation_env,
    n_eval_episodes=3,
    render=True,  # comment out to speed up
)
print(f"Reward before training: {reward}")

print("Training a policy using Behavior Cloning")
bc_trainer.train(n_epochs=1)

print("Evaluating the trained policy.")
reward, _ = evaluate_policy(
    bc_trainer.policy,  # type: ignore[arg-type]
    evaluation_env,
    n_eval_episodes=3,
    render=True,  # comment out to speed up
)
print(f"Reward after training: {reward}")