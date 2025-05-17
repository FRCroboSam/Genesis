import argparse
import os
import pickle

import torch
from go2_env import FrankaGo2Env

import genesis as gs
from scipy.spatial.transform import Rotation as R


"""
Mapping from Franka to pandas 


end_effeector -> franka.get_link("hand")

observation space
- end effector (x, y, z) -> self.franka.get_link("hand")
- block (x, y, z)        -> self.block.get_pos()
- relative block (x, y, z) to gripper
- joint displacement of of the right gripper finger ->  franka.get_dofs_position([dofs_idx[8]])
- joint displacement of the left gripper finger -> franka.get_dofs_position([dofs_idx[7]])
- block rotation in xyz, euler frame rotation (angle rad) ->
        quat = cube.get_quat()
        r = R.from_quat(quat)
        euler = r.as_euler('xyz', degrees=False)  # radians
- block linear velocity in wrt gripper -> self.cube.get_vel() - end_effector.get_vel()
- block angular velocity (x, y, z) -> self.cube.get_ang()
- end effector linear velocity (x, y, z) direction -> end_effector.get_vel()
- right gripper finger linear velocity -> franka.get_dofs_velocity([dofs_idx[8]])
- left gripper finger linear velocity -> franka.get_dofs_velocity([dofs_idx[7]])

desired goal 
- final goal block position (x, y, z)      -> self.block.get_pos()



achieved goal
- current block position (x, y, z)         -> self.cube.get_pos()


reward
- torch.norm(block pos - goal pos )

"""


TODO: test out the setup configurations for reset env and then run it in grasp cube

# https://github.com/google-deepmind/mujoco_menagerie/blob/main/franka_emika_panda/panda.xml
def get_cfgs():
    env_cfg = {
        "num_actions": 4,
        # TODO: FIND THE CORRECT VALUES FOR THIS -> Try xml file 
#           franka_pos = torch.tensor([-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.0, 0.0]).to(self.device)

        "default_joint_angles": {  # [rad]
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.8,
            "joint6": 0.8,
            "finger_joint1": 1.0,
            "finger_joint2": 1.0,
        },
        "joint_names": [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7"
            "finger_joint1",
            "finger_joint2",
        ],
        # PD
        "kp": 70.0,
        "kd": 3.0,
        # termination
        "termination_if_roll_greater_than": 1000,  # degree
        "termination_if_pitch_greater_than": 1000,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.35],
        "base_init_quat": [0.0, 0.0, 0.0, 1.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.5,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 60,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
            "reward_scales": {
            "goal_distance": -1.0,
        },
    }
    command_cfg = {
        "num_commands": 4,
        "lin_vel_x_range": [0, 0],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg




class FrankaEnv(FrankaGo2Env):
    def get_observations(self):
        self.obs_buf = torch.cat(
            [
                torch.tensor(self.franka.get_link("hand").get_pos(), dtype=torch.float32),  # end effector pos (3)
                torch.tensor(self.cube.get_pos(), dtype=torch.float32),                     # cube pos (3)
                torch.tensor(self.cube.get_pos() - self.franka.get_link("hand").get_pos(), dtype=torch.float32),  # relative cube pos (3)
                torch.tensor(self.franka.get_dofs_position([self.dofs_idx[8]]), dtype=torch.float32),  # right finger pos (1)
                torch.tensor(self.franka.get_dofs_position([self.dofs_idx[7]]), dtype=torch.float32),  # left finger pos (1)
                torch.tensor(R.from_quat(self.cube.get_quat()).as_euler('xyz', degrees=False), dtype=torch.float32),  # cube euler (3)
                torch.tensor(self.cube.get_vel() - self.franka.get_link("hand").get_vel(), dtype=torch.float32),  # relative vel (3)
                torch.tensor(self.cube.get_ang(), dtype=torch.float32),                      # cube angular vel (3)
                torch.tensor(self.franka.get_link("hand").get_vel(), dtype=torch.float32),  # end effector vel (3)
                torch.tensor(self.franka.get_dofs_velocity([self.dofs_idx[8]]), dtype=torch.float32),  # right finger vel (1)
                torch.tensor(self.franka.get_dofs_velocity([self.dofs_idx[7]]), dtype=torch.float32),  # left finger vel (1)
                torch.tensor(self.block.get_pos(), dtype=torch.float32),                    # desired goal (3)
                torch.tensor(self.cube.get_pos(), dtype=torch.float32),                     # achieved goal (3)
            ],
            dim=-1,
        )
        return self.obs_buf

    def step(self, actions):
        super().step(actions)
        self.get_observations()
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="single")
    args = parser.parse_args()

    gs.init()

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

    if args.exp_name == "single":
        env_cfg["episode_length_s"] = 2
    elif args.exp_name == "double":
        env_cfg["episode_length_s"] = 3
    else:
        raise RuntimeError

    env = FrankaEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    policy = torch.jit.load(f"./franka/{args.exp_name}.pt")
    policy.to(device=gs.device)

    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go2_backflip.py -e single
python examples/locomotion/go2_backflip.py -e double
"""
