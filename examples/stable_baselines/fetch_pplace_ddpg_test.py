import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

save_every_timesteps = 1_000_000
n_envs = 2000
save_freq = save_every_timesteps // n_envs
checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path='./models/',
                                         name_prefix='franka_checkpoint_ddpg_pickplace')
# Parallel environments
gym.register_envs(gymnasium_robotics)
vec_env = make_vec_env("FetchPickAndPlaceDense-v4", n_envs=2000)

model = DDPG("MultiInputPolicy", vec_env, verbose=1)
model.learn(total_timesteps=100000000, callback=checkpoint_callback)
model.save("fetch-pplace-gym")

# del model # remove to demonstrate saving and loading

eval_env = make_vec_env("FetchPickAndPlaceDense-v4", n_envs=1)

# model = DDPG.load("ppo-fetchpick-gym")


#This is the best model for fetch reach
model = DDPG.load("fetch-pplace-gym")

obs = eval_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated = eval_env.step(action)    
    eval_env.render("human")




#TODO: see how good this thing is if it runs, then continue getting genesis to work with stable baselines, fix step info keys reward, timesteps
#   figure out how to make multiple envs work within genesis/ test out naive through stable first