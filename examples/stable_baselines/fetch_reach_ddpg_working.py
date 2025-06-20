import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
gym.register_envs(gymnasium_robotics)
vec_env = make_vec_env("FetchReachDense-v4", n_envs=1)

model = DDPG("MultiInputPolicy", vec_env, verbose=1)      #usually
model.learn(total_timesteps=1000)
model.save("ppo-fetchpick-gym")

# del model # remove to demonstrate saving and loading

eval_env = make_vec_env("FetchReachDense-v4", n_envs=1)
print(eval_env.action_space)
print(eval_env.observation_space)
# model = DDPG.load("ppo-fetchpick-gym")

#TODO: Figure out how they configure reward for this 


#This is the best model for fetch reach
model = DDPG.load("stable_baselines/best-fetch-reach-gym")

obs = eval_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated = eval_env.step(action)    
    eval_env.render("human") 