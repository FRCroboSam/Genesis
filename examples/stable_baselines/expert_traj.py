from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.env_util import make_vec_env
from franka_env_gym import GymFrankaEnv  # assuming your wrapper is saved here
import genesis as gs
import argparse
import os
import pickle
import shutil
from importlib import metadata
from franka_env_vec import FrankaGenesisVecEnv
from franka_env_vec_ddpg import FrankaGenesisVecEnvMultiInput
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback

import numpy as np

def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 50,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 4,
        # TODO: FIND THE CORRECT VALUES FOR THIS -> Try xml file 
#           franka_pos = torch.tensor([-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.0, 0.0]).to(self.device)

        "default_joint_angles": {  # [rad]
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
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
            "finger_joint1",
            "finger_joint2",
        ],
        # PD
        "kp": 70.0,
        "kd": 3.0,
        # termination

        # base pose

        "episode_length_s": 1.0,
        "resampling_time_s": None,
        "action_scale": 0.05,
        "simulate_action_latency": False,   #can try turning this to True
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
            # "goal_distance": 1.0,
            # "lifting_block": 5,
            # "grasping_block":1.0
            "naive_distance": 1.0
            

        },
    }
    command_cfg = {
        "num_commands": 4,
        "lin_vel_x_range": [0, 0],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


parser = argparse.ArgumentParser()
parser.add_argument("-e", "--exp_name", type=str, default="franka-pick-place")
parser.add_argument("-B", "--num_envs", type=int, default=100)
parser.add_argument("-v", "--show_viewer", action="store_true", help="Show the viewer")
parser.add_argument("--max_iterations", type=int, default=1000)          #Normal is 101
args = parser.parse_args()
env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()


train_cfg = get_train_cfg(args.exp_name, args.max_iterations)
checkpoint_callback = CheckpointCallback(save_freq=1000000 // args.num_envs, save_path='./models/',
                                         name_prefix='franka_checkpoint_ddpg_pickplace')

from collections import Counter

def main():
    # eval_callback = EvalCallback(
    #     eval_env,
    #     best_model_save_path='./models/',
    #     log_path='./logs/',
    #     eval_freq=1000,
    #     deterministic=True,
    #     render=False
    # )
    observations = []
    actions = []
    dones = []

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    print("Logging for: " + str(log_dir))
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    # Create a vectorized environment
    env = FrankaGenesisVecEnv(num_envs=1, env_cfg=env_cfg, is_expert=True, obs_cfg=obs_cfg, place_only=True, reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=True)
    
    obs = env.reset()
    for _ in range(500):
        actual_obs, action, dones, successes = env.env.expert_step(obs)
        observations.append(obs)
        actions.append(action)
        obs = actual_obs
    print(actions)


    
    # Suppose actions is an ndarray of shape (N, action_dim)
    actions_tuples = [tuple(action.flatten()) for action in actions]

    counts = Counter(actions_tuples)
    num_duplicates = sum(1 for count in counts.values() if count > 1)
    print(f"Number of duplicated actions: {num_duplicates}")


    eval_env = env
    # eval_env = FrankaGenesisVecEnv(num_envs=1, env_cfg=env_cfg, is_expert=True, obs_cfg=obs_cfg, place_only=True, reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=True)



    #TODO: pass in the corresponding obs as well
    obs = eval_env.reset()
    for i in range(500):
        action = actions[i]
        obs = observations[i]
        # print("ACTION: " + str(action))
        obs, action, reward, done, info = eval_env.env.expert_step_with_rew_action(obs, action) #eval_env.step(action)
        actions[i] = action



    print("3rd Round")
    # #DO IT FOR REAL THIS TIME
    obs = eval_env.reset()
    for i in range(500):
        action = actions[i]
        print("ACTION: " + str(action))
        goal = observations[i][:, 28:31]    #TODO FIGURE THIS OUT MAKE THIS WORK

        # print("ACTION: " + str(action))
        obs, action, reward, done, info = eval_env.env.expert_step_with_rew_action(obs, action, goal) #eval_env.step(action)


#TODO: compare these optimal actions with the actions of the traine policy and opt poli8cy: if they are similar something wrong with env space

if __name__ == "__main__":
    main()

