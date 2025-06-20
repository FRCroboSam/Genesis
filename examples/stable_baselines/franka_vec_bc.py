import numpy as np
import argparse
import os
import pickle
import torch
from stable_baselines3 import DDPG, PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from imitation.algorithms import bc
from imitation.data.types import TrajectoryWithRew
from imitation.util.util import make_vec_env as make_vec_env_imitation
import genesis as gs
from franka_env_vec import FrankaGenesisVecEnv
from franka_env_vec_ddpg import FrankaGenesisVecEnvMultiInput
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.data import rollout


def get_cfgs():
    env_cfg = {
        "num_actions": 4,
        "default_joint_angles": {
            "joint1": -1.0124,
            "joint2": 1.5559,
            "joint3": 1.3662,
            "joint4": -1.6878,
            "joint5": -1.5799,
            "joint6": 1.7757,
            "joint7": 1.4602,
            "finger_joint1": 0.0,
            "finger_joint2": 0.0,
        },
        "joint_names": [
            "joint1", "joint2", "joint3", "joint4", "joint5", 
            "joint6", "joint7", "finger_joint1", "finger_joint2"
        ],
        "kp": 70.0,
        "kd": 3.0,
        "episode_length_s": 1.0,
        "action_scale": 0.05,
        "simulate_action_latency": False,
        "clip_actions": 1.0,
    }
    obs_cfg = {
        "num_obs": 35,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "reward_scales": {
            "naive_distance": 1.0,
        },
    }
    command_cfg = {
        "num_commands": 4,
        "lin_vel_x_range": [0, 0],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg


from imitation.data.types import Transitions

from imitation.data.types import TrajectoryWithRew
import numpy as np

def collect_trajectories(env, num_steps=500, traj_len=51):
    torch.set_default_device('cuda')
    trajectories = []
    obs = env.reset()
    
    def to_np(x):
        return x.cpu().numpy() if hasattr(x, "cpu") else np.array(x)
    
    obs_list, act_list, rews_list, done_list, info_list = [], [], [], [], []
    obs_list.append(to_np(obs))
    steps_collected = 0
    while steps_collected < num_steps:
        action, dones, _, rew = None, None, None, None
        # Record current obs


        # Expert step function call
        # Assuming expert_step_with_rew returns (next_obs, action, dones, successes, rew)
        next_obs, action, dones, successes, rew = env.env.expert_step_with_rew(obs)

        act_list.append(to_np(action))
        rews_list.append(to_np(rew))
        done_list.append(to_np(dones))
        info_list.append({})  # add actual info if available
        
        obs = next_obs
        obs_np = to_np(obs)
        obs_list.append(obs_np)
        steps_collected += 1

        print("OBS LIST LENGTH: " + str(len(obs_list)))
        print("ACTION LIST LENGTH: " + str(len(act_list)))
        # When trajectory length reached or done signal, finalize trajectory
        if (steps_collected % traj_len == 0) or dones:
            print("ADDING A TRAJECTORY")
            traj = TrajectoryWithRew(
                obs=np.array(obs_list).squeeze(),
                acts=np.array(act_list).squeeze(),
                rews=np.array(rews_list).squeeze(),
                infos=info_list,
                terminal=bool(dones),
            )
            trajectories.append(traj)

            # reset buffers for next trajectory
            obs_list, act_list, rews_list, done_list, info_list = [], [], [], [], []

            obs = env.reset()

            # Include the last observation of the new trajectory as start of next
            if obs is not None:
                obs_list.append(to_np(obs))

    # Handle any leftover steps after loop ends (optional)
    if len(obs_list) > 1:
        traj = TrajectoryWithRew(
            obs=np.array(obs_list).squeeze(),
            acts=np.array(act_list).squeeze(),
            rews=np.array(rews_list).squeeze(),
            infos=info_list,
            terminal=bool(done_list[-1]) if done_list else False,
        )
        trajectories.append(traj)

    return trajectories




def train_bc(env, transitions, observation_space, action_space):

    transitions = rollout.flatten_trajectories(transitions)
    print("TRANSITIONS: " + str(transitions))
    """Train behavior cloning on collected transitions."""
    bc_trainer = bc.BC(
        observation_space=observation_space,
        action_space=action_space,
        demonstrations=transitions,
        rng=np.random.default_rng(0),
    )

    print("OBSERVATION SPACE: " + str(observation_space))
    print("ACTION SPACE: " + str(action_space))
    print("DEMONSTRATIONS: " + str(transitions))


    devices = {param.device for param in bc_trainer.policy.parameters()}



    print(f"BC policy parameters devices: {devices}")
    # bc_trainer._generator = torch.Generator(device="cuda")    
    # print("BC Generator device:", bc_trainer._generator.device)
    print("[BC] Evaluating before training...")
    reward, _ = evaluate_policy(bc_trainer.policy, env, n_eval_episodes=5)
    print(f"[BC] Reward before training: {reward}")

    bc_trainer.train(n_epochs=5)

    print("[BC] Evaluating after training...")
    reward, _ = evaluate_policy(bc_trainer.policy, env, n_eval_episodes=5)
    print(f"[BC] Reward after training: {reward}")

    return bc_trainer.policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="franka-pick-place")
    parser.add_argument("--max_iterations", type=int, default=1000)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    env = FrankaGenesisVecEnv(num_envs=1, env_cfg=env_cfg, is_expert=True,
                              obs_cfg=obs_cfg, place_only=True, reward_cfg=reward_cfg,
                              command_cfg=command_cfg, show_viewer=True)

    print("[INFO] Collecting expert demonstrations...")
    transitions = collect_trajectories(env)

    print("[INFO] Training BC model...")
    pretrained_policy = train_bc(env, transitions, env.observation_space, env.action_space)

    print("[INFO] Creating multi-input env for DDPG...")
    ddpg_env = FrankaGenesisVecEnv(num_envs=1, env_cfg=env_cfg,
                                             obs_cfg=obs_cfg, place_only=True, reward_cfg=reward_cfg,
                                             command_cfg=command_cfg, show_viewer=True)

    # Initialize DDPG with pretrained actor weights from BC (if architecture matches)
    model = PPO(
        MlpPolicy,  # You may need to customize this
        ddpg_env,
        # batch_size=32,
        policy_kwargs=dict(net_arch=[32, 32]),  # <-- match BC
        verbose=1,
        tensorboard_log="./ddpg_bc_logs/",
    )

    # [OPTIONAL] Transfer weights from BC to DDPG actor (requires custom logic)
    model.policy.load_state_dict(pretrained_policy.state_dict())

    print("[INFO] Fine-tuning with DDPG...")
    model.learn(total_timesteps=100_000)


if __name__ == "__main__":
    main()


#TODO: see if you can visualize the eval performance of the expert trajectories















# 
# 
# 
#  import numpy as np
# from stable_baselines3 import PPO
# from stable_baselines3.ppo import MlpPolicy
# from stable_baselines3.common.evaluation import evaluate_policy
# import gymnasium_robotics

# from imitation.algorithms import bc
# from imitation.data import rollout
# from imitation.data.types import TrajectoryWithRew
# from imitation.util.util import make_vec_env
# import gymnasium as gym
# import gymnasium_robotics
# from stable_baselines3 import DDPG
# from stable_baselines3.common.env_util import make_vec_env

# # Parallel environments
# gym.register_envs(gymnasium_robotics)
# vec_env = make_vec_env("FetchReachDense-v4", n_envs=1)
# env = vec_env


# # STEP 2: COLLECT EXPERT DEMONSTRATIONS
# # You can train your own expert or load one.
# # Here we just train a quick PPO policy to act as the expert.
# expert = PPO(policy=MlpPolicy, env=vec_env, verbose=0)
# expert.learn(total_timesteps=5000)

# # Use rollout to generate expert demonstrations
# rollouts = rollout.rollout(
#     expert,
#     vec_env,
#     rollout.make_sample_until(min_episodes=10),
#     rng=np.random.default_rng(0),
# )
# transitions = rollout.flatten_trajectories(rollouts)

# # STEP 3: TRAIN WITH BEHAVIOR CLONING
# bc_trainer = bc.BC(
#     observation_space=env.observation_space,
#     action_space=env.action_space,
#     demonstrations=transitions,
#     rng=np.random.default_rng(0),
# )

# # Evaluate before BC
# reward, _ = evaluate_policy(bc_trainer.policy, env, n_eval_episodes=5)
# print(f"[BC] Reward before training: {reward}")

# # Train using Behavior Cloning
# bc_trainer.train(n_epochs=5)

# # Evaluate after BC
# reward, _ = evaluate_policy(bc_trainer.policy, env, n_eval_episodes=5)
# print(f"[BC] Reward after training: {reward}")

# # STEP 4: CONTINUE TRAINING WITH PPO USING THE BC POLICY
# ppo = PPO(
#     policy=bc_trainer.policy,  # Use the pretrained policy
#     env=env,
#     verbose=1,
# )

# # Train further with RL
# ppo.learn(total_timesteps=100_000)

# # Evaluate final PPO policy
# reward, _ = evaluate_policy(ppo.policy, eval_env, n_eval_episodes=5)
# print(f"[PPO] Reward after PPO fine-tuning: {reward}")
