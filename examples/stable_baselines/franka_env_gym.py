import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from franka_env_genesis import FrankaEnvGenesis
class GymFrankaEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_envs=1, env_cfg=None, obs_cfg=None, reward_cfg=None, command_cfg=None, place_only=False):
        super(GymFrankaEnv, self).__init__()
        self.env = FrankaEnvGenesis(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, place_only=place_only)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.env.obs_buf.shape[1],), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.env.num_actions,), dtype=np.float32)
        self._obs = None

    def reset(self):
        obs, _ = self.env.reset()
        self._obs = obs[0].cpu().numpy()
        return self._obs

    def step(self, action):
        action_tensor = torch.tensor(action, device=self.env.device).unsqueeze(0)
        obs, reward, done, info = self.env.step(action_tensor)
        self._obs = obs[0].cpu().numpy()
        reward = reward[0].item()
        done = bool(done[0].item())
        info_dict = {k: v[0].item() if isinstance(v, torch.Tensor) else v for k, v in info.items()}
        return self._obs, reward, done, info_dict

    def render(self, mode='human'):
        pass  # already handled internally if show_viewer=True

    def close(self):
        del self.env
