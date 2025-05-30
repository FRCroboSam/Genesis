import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
import numpy as np
from scipy.spatial.transform import Rotation as R
from numpy import random 

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class FrankaGo2Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, place_only=False):
        self.place_only = place_only
        print(f"Place Only{self.place_only}")
        self.goal_index = 0
        self.target_poses = []
        self.reach_target_threshold = 0.08
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add plain
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        # self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file="../assets/xml/franka_emika_panda/panda.xml"),
        )
        
        self.end_effector = self.franka.get_link("hand")
        if not self.place_only:
            self.cube = self.scene.add_entity(
                gs.morphs.Box(
                    size=(0.04, 0.04, 0.04), # block
                    pos=(0.66, 0.0, 0.02),
                )
            )
        else:
            self.cube = self.scene.add_entity(
                gs.morphs.Box(
                    size=(0.07, 0.07, 0.07), # block
                    pos=(0.66, 0.0, 0.05),
                    euler=(0, 0, 0)
                )
            )
        
        self.envs_idx = np.arange(num_envs)

        

        
        self.goal_target = self.scene.add_entity(
            gs.morphs.Sphere(
                pos=(0.0, 0.0, 0.0),
                euler=(0.0, 0.0, 0.0),
                visualization=True,
                collision=False,
                requires_jac_and_IK=False,
                fixed=True,
                radius=0.04
            )
        )
        self.default_goal_pos = np.array([0.7, 0.0, 0])

        # Initialize random goal target positions
        for _ in range(12):
            # default range
            offset = np.array([random.rand() * 0.2, random.rand() * 0.6 - 0.3, 0.35 * random.rand() + 0.1])
            #less picky range
            # offset = np.array([random.rand() * 0.1, random.rand() * 0.4 - 0.2, 0.2 * random.rand() + 0.1])

            target_pos = self.default_goal_pos + offset
            target_pos = np.repeat(target_pos[np.newaxis], self.num_envs, axis=0)
            self.target_poses.append(target_pos)



        
        #TODO: CONTINUE FIXING THIS

        # build
        self.scene.build(n_envs=num_envs)
        
        pos = torch.tensor([ 0.6781, -0.0205,  0.3626], dtype=torch.float32, device=self.device)
        self.pos = pos.unsqueeze(0).repeat(self.num_envs, 1)
        quat = torch.tensor([0, 1, 0, 0], dtype=torch.float32, device=self.device)
        self.quat = quat.unsqueeze(0).repeat(self.num_envs, 1)
        
        self.motors_dof = torch.arange(7).to(self.device)
        self.fingers_dof = torch.arange(7, 9).to(self.device)
        
        

        # names to indices
        print("ENV CONFIG: " + str(env_cfg))
        for name in env_cfg["joint_names"]:
            print("JOINT NAME IS: " + name)
        self.dofs_idx = [self.franka.get_joint(name).dof_idx_local for name in env_cfg["joint_names"]]    


        # PD control parameters
        # self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        # self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # initialize buffers
        
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()
        self.reset()

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), gs.device)

    def step(self, actions):
        # print("TIMESTEP: " + str(self.episode_length_buf))
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        
        if self.place_only:
            exec_actions[:,3] = 1 #-> This hard codes it to close
        delta_pos = exec_actions[:, :3] * 0.05  #should be 5cm max movement
        gripper_cmd = exec_actions[:, 3]



        finger_width = (1 - gripper_cmd) * 0.02  # Map [-1,1]→[0,0.04]
        finger_pos = torch.stack([finger_width, finger_width], dim=1)  # Both fingers
        self.pos += delta_pos




        self.qpos = self.franka.inverse_kinematics(
            link=self.franka.get_link("hand"),
            pos=self.pos,
            quat=self.quat,
        )


        # # Execute movements
        self.franka.control_dofs_position(self.qpos[:, :-2], self.motors_dof, self.envs_idx)

        # if not self.place_only:
        self.franka.control_dofs_position(finger_pos, self.fingers_dof, self.envs_idx)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1



        # resample commands
        # envs_idx = (
        #     (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
        #     .nonzero(as_tuple=False)
        #     .flatten()
        # )

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length

        self.reset_buf |= self._goal_distance() <= self.reach_target_threshold


        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            # print("REWARD FUNC IS: " + str(reward_func))
            rew = reward_func() * self.reward_scales[name]
            # print("REWARD IS: " + str(rew))
            self.rew_buf += rew
            self.episode_sums[name] += rew
        
        cube_euler = torch.tensor(R.from_quat(self.cube.get_quat().detach().cpu().numpy()).as_euler('xyz', degrees=False), dtype=torch.float32)

        # compute observations
        self.obs_buf = torch.cat(
            [
                torch.tensor(self.franka.get_link("hand").get_pos(), dtype=torch.float32),  # end effector pos (3)
                torch.tensor(self.cube.get_pos(), dtype=torch.float32),                     # cube pos (3)
                torch.tensor(self.cube.get_pos() - self.franka.get_link("hand").get_pos(), dtype=torch.float32),  # relative cube pos (3)
                torch.tensor(self.franka.get_dofs_position([self.dofs_idx[8]]), dtype=torch.float32),  # right finger pos (1)
                torch.tensor(self.franka.get_dofs_position([self.dofs_idx[7]]), dtype=torch.float32),  # left finger pos (1)
                cube_euler,                                                                 # cube euler (3)
                torch.tensor(self.cube.get_vel() - self.franka.get_link("hand").get_vel(), dtype=torch.float32),  # relative vel (3)
                torch.tensor(self.cube.get_ang(), dtype=torch.float32),                      # cube angular vel (3)
                torch.tensor(self.franka.get_link("hand").get_vel(), dtype=torch.float32),  # end effector vel (3)
                torch.tensor(self.franka.get_dofs_velocity([self.dofs_idx[8]]), dtype=torch.float32),  # right finger vel (1)
                torch.tensor(self.franka.get_dofs_velocity([self.dofs_idx[7]]), dtype=torch.float32),  # left finger vel (1)
                torch.tensor(self.goal_target.get_pos(), dtype=torch.float32),                    # desired goal (3)
                torch.tensor(self.cube.get_pos(), dtype=torch.float32),                     # achieved goal (3)
                self.actions
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    #TODO: make this mimic franka pick place 
    # generate the random location for the task
    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        # original pos for right on top of cube -> good for model_100 best with 0.66og
        if not self.place_only:
            franka_pos = torch.tensor([-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04]).to(self.device)


        #TODO: make it work with this slightly harder starting pos
        # franka_pos = torch.tensor([-1.0124, 1.5559, 1.4662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04]).to(self.device)

        # franka_pos = torch.tensor(
        #     [-1.075, 1.5559, 1.7662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04], 
        #     device=self.device
        # )    -> modified harder position 

        else:     #place_only case
            franka_pos = torch.tensor([-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.0, 0.0]).to(self.device)


        #doing set pos doesn't actually move it
        # print("INITIAL FRANKA POS: " + str(self.franka.get_link("hand").get_pos()))    #THis gets you accurate position -> based on qpos you set earlier
        franka_pos = franka_pos.unsqueeze(0).repeat(len(envs_idx), 1)
        self.franka.set_qpos(franka_pos, envs_idx=envs_idx)
        self.scene.step()

        # Reset pos and quat only for the envs being reset
        # pos = torch.tensor([], dtype=torch.float32, device=self.device)
        # self.pos[envs_idx] = pos.unsqueeze(0).repeat(len(envs_idx), 1)

        quat = torch.tensor([0, 1, 0, 0], dtype=torch.float32, device=self.device)
        self.quat[envs_idx] = quat.unsqueeze(0).repeat(len(envs_idx), 1)

        # Reset cube position only for envs being reset
        if not self.place_only:
            cube_pos = np.array([0.66, 0.0, 0.02])
        else:
            cube_pos = np.array([0.65, 0.0, 0.05])

        cube_pos_batch = np.repeat(cube_pos[np.newaxis], len(envs_idx), axis=0)

        self.cube.set_pos(cube_pos_batch, envs_idx=envs_idx)

        # Reset arm pos for envs being reset -> unused
        arm_pos = torch.tensor([0.1, 0.0, 0.8], device=self.device)
        arm_pos = arm_pos.unsqueeze(0).repeat(len(envs_idx), 1)
        # (You can assign/use arm_pos here as needed for envs_idx)

        # Generate random offset per env to reset
        x = torch.empty(len(envs_idx), device=self.device).uniform_(0.0, 0.2)
        y = torch.empty(len(envs_idx), device=self.device).uniform_(-0.3, 0.3)
        z = torch.empty(len(envs_idx), device=self.device).uniform_(0.1, 0.45)

        offsets = torch.stack([x, y, z], dim=1)  # shape: (len(envs_idx), 3)

        # Convert default_goal_pos to a torch tensor and repeat for envs_idx length
        default_goal_pos = torch.tensor(self.default_goal_pos, dtype=torch.float32, device=self.device)
        default_goal_pos = default_goal_pos.unsqueeze(0).repeat(len(envs_idx), 1)  # shape: [len(envs_idx), 3]

        new_goal_pos = default_goal_pos + offsets  # both torch tensors on the same device
        # print("NEW GOAL POS FOR: " + str(envs_idx))
        self.goal_target.set_pos(new_goal_pos, envs_idx=envs_idx)
        # print("SELF GOAL TARGET POSES ARE: " + str(self.goal_target.get_pos()))

        # Reset buffers only for envs_idx
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # Update extras for envs_idx
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        # Uncomment if you want to resample commands
        # self._resample_commands(envs_idx)

        # for _ in range(10):  # or however many extra steps you want
        #     self.scene.step()


    # def reset_idx(self, envs_idx):
    #     if len(envs_idx) == 0:
    #         return

    #     # reset dofs
    #     # print("DOF POS SHAPE: " + str(self.dof_pos.shape))
    #     # print("DEFAULT DOF POS: " + str(self.default_dof_pos))
    #     # self.dof_pos[envs_idx] = self.default_dof_pos
    #     # self.dof_vel[envs_idx] = 0.0


    #     franka_pos = torch.tensor([-1.075, 1.5559, 1.7662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04]).to(self.device)

    #     franka_pos = franka_pos.unsqueeze(0).repeat(len(envs_idx), 1)  # repeat only for envs being reset
    #     self.franka.set_qpos(franka_pos, envs_idx=envs_idx)
    #     self.scene.step()

    #     # Initial end effector target original 0.135
    #     # pos = torch.tensor([1.65, -1.2, 0.135], dtype=torch.float32, device=self.device)
    #     pos = torch.tensor([0, 0, 0], dtype=torch.float32, device=self.device)
    #     self.pos = pos.unsqueeze(0).repeat(self.num_envs, 1)
        
        
    #     quat = torch.tensor([0, 1, 0, 0], dtype=torch.float32, device=self.device)
    #     self.quat = quat.unsqueeze(0).repeat(self.num_envs, 1)
        
    #     cube_pos = np.array([0.65, 0.0, 0.02])
    #     cube_pos = np.repeat(cube_pos[np.newaxis], self.num_envs, axis=0)
    #     self.cube.set_pos(cube_pos, envs_idx=self.envs_idx)
        

    #     arm_pos = torch.tensor([0.1, 0.0, 0.8]).to(self.device)
    #     arm_pos = arm_pos.unsqueeze(0).repeat(len(envs_idx), 1)  # repeat only for envs being reset




    #     x = np.random.uniform(0.0, 0.2)       # [0, 0.2]
    #     y = np.random.uniform(-0.3, 0.3)      # [-0.3, 0.3]
    #     z = np.random.uniform(0.1, 0.45)      # [0.1, 0.45]

    #     offset = np.array([x, y, z])

    #     new_goal_pos = self.default_goal_pos + offset
    #     new_goal_pos = new_goal_pos.unsqueeze(0).repeat(self.envs_idx)

    #     self.goal_target.set_pos(new_goal_pos, envs_idx=self.envs_idx)  #we already did the repeat earlier

    #     # reset buffers
    #     self.last_actions[envs_idx] = 0.0
    #     self.last_dof_vel[envs_idx] = 0.0
    #     self.episode_length_buf[envs_idx] = 0
    #     self.reset_buf[envs_idx] = True

    #     # fill extras
    #     self.extras["episode"] = {}
    #     for key in self.episode_sums.keys():
    #         self.extras["episode"]["rew_" + key] = (
    #             torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
    #         )
    #         self.episode_sums[key][envs_idx] = 0.0

    #     # self._resample_commands(envs_idx)

    #     # self.franka.set_pos(self.pos, envs_idx=envs_idx)
    #     # self.scene.step()

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    
    # reward based on how close cube is to the goal target
    # reward scales make this negative later

    def _reward_grasping_block(self):
        # Midpoint between left and right fingers
        finger_pos = (self.franka.get_link("right_finger").get_pos() + self.franka.get_link("left_finger").get_pos()) / 2.0

        # Distance between cube and finger midpoint
        distance = torch.norm(self.cube.get_pos() - finger_pos, dim=1)

        # Thresholds
        best_dist = 0.05      # Best proximity (reward = +1)
        close_dist = 0.10     # Within grasping range (small positive)
        max_dist = 0.25       # Beyond this: strongly negative

        reward = torch.zeros_like(distance)

        # Case 1: distance <= close_dist → linearly positive up to +1 at best_dist
        close_mask = distance <= close_dist
        reward[close_mask] = (close_dist - distance[close_mask]) / (close_dist - best_dist)

        # Case 2: distance > close_dist → linearly decreasing negative reward
        far_mask = distance > close_dist
        reward[far_mask] = - (distance[far_mask] - close_dist) / (max_dist - close_dist)

        # Optional: Clamp to avoid large values
        reward = torch.clamp(reward, -1.0, 1.0)

        # print("GRASP BLOCK DIST:", distance)
        # print("REWARD:", reward)

        gripper_closed = self.franka.get_dofs_position([self.dofs_idx[8]]) <= 0.02

        reward = torch.clamp(reward, -1.0, 1.0)

        # Check if gripper is closed (assuming right finger is sufficient)
        gripper_closed = self.franka.get_dofs_position([self.dofs_idx[8]]) <= 0.02  # shape: [batch_size, 1]

        # Add +2 bonus where both gripper is closed AND object is within close_dist
        bonus_mask = torch.logical_and(gripper_closed.squeeze(), close_mask)
        reward[bonus_mask] += 2.0

        return reward




    def _reward_pick_cube(self):
        gripper_position = (self.franka.get_link("left_finger").get_pos() + self.franka.get_link("right_finger").get_pos()) / 2        
        # gripper_height = gripper_position[:, 2]

        # Clamp block Z to [0.02, 0.2] for reward
        block_lift_reward = torch.clamp(self.cube.get_pos()[:, 2], min=0.02, max=0.2) * 10

        # Penalize lifting gripper above 0.3 (subtract penalty if height > 0.3)
        # gripper_penalty = torch.clamp(gripper_height - 0.3, min=0.0) * 10.0  # penalty scale = 10.0
        reward = (
            -torch.norm(self.cube.get_pos() - gripper_position, dim=1)
            + block_lift_reward
            # - gripper_penalty
        )

        return reward
    


    def _reward_place_cube(self):
        distance = torch.norm(self.cube.get_pos() - self.goal_target.get_pos(), dim=1)
        cube_arm_dist = -torch.norm(self.cube.get_pos() - self.end_effector.get_pos(), dim=1) * 5
        return -distance + cube_arm_dist


    #TODO RUN THIS FOR LONGER WITH MORE ENVS
    def _reward_lifting_block(self):
        reward = torch.maximum(torch.tensor(0.0), self.cube.get_pos()[:, 2] - 0.0199) * 20
        # print("CUBE HEIGHT: " + str(self.cube.get_pos()[:, 2]) + " REWARD: " + str(reward))

        return reward

    def _reward_goal_distance(self):
        # Compute distance between cube and goal
        distance = torch.norm(self.cube.get_pos() - self.goal_target.get_pos(), dim=1)

        # Define thresholds
        max_dist = 0.4    # Worst case
        zero_point = 0.2  # Reward = 0 here
        best_dist = 0.05  # Best case (max reward here)

        # Initialize reward tensor
        reward = torch.zeros_like(distance)

        # Case 1: distance > 0.2 -> linearly negative reward
        far_mask = distance > zero_point
        reward[far_mask] = - (distance[far_mask] - zero_point) / (max_dist - zero_point)

        # Case 2: distance <= 0.2 -> linearly increasing reward toward 0.05
        close_mask = distance <= zero_point
        reward[close_mask] = (zero_point - distance[close_mask]) / (zero_point - best_dist)

        # print("GOAL DISTANCE:" + str(distance) + " reward: " + str(reward))

        return reward

    def _goal_distance(self):
        return torch.norm(self.cube.get_pos() - self.goal_target.get_pos(), dim=1)
    




#TODO  MAKE BETTER -> reward shaping?
#   see if it learns faster with pick only if i make it easier -> move the arm back to really close to the cube, like 0.65 or something

#wokflow -> change some params, test with 1 env viz see if it looks ok then start training



#STEPS: Run rochelle Ni's grasp cube, see if you can get it to work normally with the current param ie. move arm up and then if you can't
#   lower the arm pos again and then if that still doesnt work try to get genesis env repo to work using a continuous action space

#if you can make that work make this work for train_pick_only
