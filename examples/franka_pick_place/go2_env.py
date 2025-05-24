import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
import numpy as np
from scipy.spatial.transform import Rotation as R
from numpy import random 

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

#TODO: figure out why the timestep when you actual start stepping doesn't start at 0


#TODO: FIIGURE OUT WHY ARM IS STILL MOVING WHEN I TURN OFF THE dof_position movement in step
#TODO: FIGURE OUT WHY TIMESTPE WASNT AT ZERO BEFORE
# TODO: TEST THAT ACTION ACTUALLY CORRESPONDS TO CORRECT MOVEMENT AND IS APPLIED IN THE CORRECT PLACE
# #TODO: FIGURE OUT WHAT POS ACTUALLY MEANS AND WHATS DIFF BETWEEN THAT AND THE ANGLE STUFF





class FrankaGo2Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):

        self.first_time_step = True       #TODO: FIGURE OUT THE CORE ISSUE THIS IS JUST HACK FIX FOR NOW
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
        self.dt = 0.5  # control frequency on real robot is 50hz, default 0.02
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)
        print("MAX EPISODE LENGTH: " + str(self.max_episode_length))

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        show_viewer = True

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

        
        self.cube = self.scene.add_entity(
            gs.morphs.Box(
                size=(0.04, 0.04, 0.04), # block
                pos=(0.65, 0.0, 0.02),
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
        default_goal_pos = np.array([0.7, 0.0, 0])

        # Initialize random goal target positions
        for _ in range(12):
            # default range
            offset = np.array([random.rand() * 0.2, random.rand() * 0.6 - 0.3, 0.35 * random.rand() + 0.1])
            #less picky range
            # offset = np.array([random.rand() * 0.1, random.rand() * 0.4 - 0.2, 0.2 * random.rand() + 0.1])

            target_pos = default_goal_pos + offset
            target_pos = np.repeat(target_pos[np.newaxis], self.num_envs, axis=0)
            self.target_poses.append(target_pos)



        
        #TODO: CONTINUE FIXING THIS

        # build
        self.scene.build(n_envs=num_envs)
        
        pos = torch.tensor([1.65, -1.2, 0.135], dtype=torch.float32, device=self.device)
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
        self.global_timestep = self.episode_length_buf

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
        if self.first_time_step:
            print("FIRST TIME STEP: " + "RESETTING")
            self.reset()
            self.first_time_step = False 
        print("TIMESTEP: " + str(self.episode_length_buf))
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions


        delta_pos = exec_actions[:, :3] * 0.05  #should be 5cm max movement
        gripper_cmd = exec_actions[:, 3]

        finger_width = (1 + gripper_cmd) * 0.02  # Map [-1,1]→[0,0.04]
        finger_pos = torch.stack([finger_width, finger_width], dim=1)  # Both fingers
        



        delta_pos = 0    #JUST TO SEE WHAT STARTING CONFIG ACTUALLY LOOKS LIKE
        self.pos += delta_pos

        print("SELF POS: " + str(self.pos))


        self.qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=self.pos,
            quat=self.quat,
        )


        
        # Execute movements
        # self.franka.control_dofs_position(self.qpos[:, :-2], self.motors_dof, self.envs_idx)

        # if not self.place_only:
        self.franka.control_dofs_position(finger_pos, self.fingers_dof, self.envs_idx)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.global_timestep = self.episode_length_buf



        # resample commands
        # envs_idx = (
        #     (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
        #     .nonzero(as_tuple=False)
        #     .flatten()
        # )

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length

        self.reset_buf |= self._reward_goal_distance() <= self.reach_target_threshold
        print(str("REWARD DISTANCE: " + str(self._reward_goal_distance())))


        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
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
        print("RESETTING at timestep: " + str(self.global_timestep))

        # reset dofs
        # print("DOF POS SHAPE: " + str(self.dof_pos.shape))
        # print("DEFAULT DOF POS: " + str(self.default_dof_pos))
        # self.dof_pos[envs_idx] = self.default_dof_pos
        # self.dof_vel[envs_idx] = 0.0


        franka_pos = self.default_dof_pos  # (9,)
        franka_pos = franka_pos.unsqueeze(0).repeat(len(envs_idx), 1)  # repeat only for envs being reset
        self.franka.set_qpos(franka_pos, envs_idx=envs_idx)
        self.scene.step()

        # Initial end effector target original 0.135
        pos = torch.tensor([1.65, -1.2, 0.135], dtype=torch.float32, device=self.device)
        self.pos = pos.unsqueeze(0).repeat(self.num_envs, 1)
        
        
        quat = torch.tensor([0, 1, 0, 0], dtype=torch.float32, device=self.device)
        self.quat = quat.unsqueeze(0).repeat(self.num_envs, 1)
        
        cube_pos = np.array([0.65, 0.0, 0.06])
        cube_pos = np.repeat(cube_pos[np.newaxis], self.num_envs, axis=0)
        self.cube.set_pos(cube_pos, envs_idx=self.envs_idx)
        



        goal_pos = self.target_poses[self.goal_index % len(self.target_poses)]
        self.goal_target.set_pos(goal_pos, envs_idx=self.envs_idx)  #we already did the repeat earlier
        self.goal_index += 1

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True
        print("RESETTING FOR: " + str(envs_idx) +  " with shape: " + str(envs_idx.shape))
        print("RESET BUFS ARE: " + str(self.episode_length_buf))

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        # self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        print("STARTING EPISODE LENGTH BUF: " + str(self.episode_length_buf))
        return self.obs_buf, None

    # ------------ reward functions----------------
    
    # reward based on how close cube is to the goal target
    # reward scales make this negative later
    def _reward_goal_distance(self):
        return torch.norm(self.cube.get_pos() - self.goal_target.get_pos(), dim=1)