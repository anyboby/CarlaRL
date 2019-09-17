import argparse
import os
import carla_rllib
import json
import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from carla_rllib.environments.carla_envs.config import BaseConfig, parse_json
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import VecFrameStack



# CARLA settings
# ------------------------------------------------------------------------
argparser = argparse.ArgumentParser(
    description='CARLA RLLIB ENV')
package_path, _ = os.path.split(os.path.abspath(carla_rllib.__file__))
argparser.add_argument(
    '-c', '--config',
    metavar='CONFIG',
    default=os.path.join(package_path +
                         "/config.json"),
    type=str,
    help='Path to configuration file (default: root of the package -> carla_rllib)')
args = argparser.parse_args()
config_json = json.load(open(args.config))
configs = parse_json(config_json)
print("-----Configuration-----")
print(configs[0])
# ------------------------------------------------------------------------
# Mode
# Select from: DDPG, PPO, SAC
MODE = "PPO"

# ------------------------------------------------------------------------
# Main
try:
    env = gym.make("CarlaBaseEnv-v0", config=configs[0])
    obs = env.reset()

    t = 0
    print("-----Carla Environment is running-----")
    plt.ion()
    plt.show()

    while True:

        # ------------------------------------------------------------------------
        # Stable baselines
        if MODE == "DDPG":
            from stable_baselines.ddpg.policies import CnnPolicy
            from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
            from stable_baselines import DDPG

            env = DummyVecEnv([lambda: env])
            env = VecFrameStack(env, n_stack=4)
            n_actions = env.action_space.shape[-1]
            param_noise = None
            action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.3) * np.ones(n_actions))

            model = DDPG(CnnPolicy, env, verbose=0, param_noise=param_noise, action_noise=action_noise, tensorboard_log="./tensorboard_logs/")
            model.learn(total_timesteps=400000)
            model.save("carla_ddpg")
            obs = env.reset()
            while True:
                action, _states = model.predict(obs)
                obs, rewards, dones, info = env.step(action)
                env.render()
        if MODE == "PPO": # Not working yet
            from stable_baselines.common.policies import CnnPolicy
            from stable_baselines.common.policies import MlpPolicy
            from stable_baselines.common.policies import CnnLnLstmPolicy
            from stable_baselines import PPO2
            env = DummyVecEnv([lambda: env])
            env = VecFrameStack(env, n_stack=4)
            # Allow less clipping
            # Increased learning rate
            # Faster updates
            model = PPO2(CnnPolicy, env, verbose=0, tensorboard_log="./tensorboard_logs/", learning_rate=0.001, n_steps=512, cliprange=0.1, noptepochs=2, nminibatches=1)
            model.learn(total_timesteps=400000)
            model.save("carla_ppo")
            obs = env.reset()
            while True:
                action, _states = model.predict(obs)
                obs, rewards, dones, info = env.step(action)
                env.render()
        if MODE == "SAC":
            from stable_baselines.sac.policies import CnnPolicy
            from stable_baselines import SAC
            env = VecFrameStack(env, n_stack=4)
            env = DummyVecEnv([lambda: env])
            model = SAC(CnnPolicy, env, verbose=1, tensorboard_log="./tensorboard_logs/", full_tensorboard_log=True)
            # When one episode has 1000 steps the paremter means = 50 episodes
            model.learn(total_timesteps=50000, log_interval=100)
            model.save("carla_sac")
            obs = env.reset()
            while True:
                action, _states = model.predict(obs)
                #print(action)
                obs, rewards, dones, info = env.step(action)
                env.render()
        if MODE == "A2C":

            from stable_baselines.a2c.policies import CnnPolicy
            from stable_baselines.common.vec_env import SubprocVecEnv
            from stable_baselines import A2C
            env = SubprocVecEnv([lambda: env])
            model = A2C(CnnPolicy, env, verbose=1, tensorboard_log="./tensorboard_logs/", full_tensorboard_log=True)
            # When one episode has 1000 steps the paremter means = 50 episodes
            model.learn(total_timesteps=10000, log_interval=10)
            model.save("carla_a2c")
            obs = env.reset()
            while True:
                action, _states = model.predict(obs)
                #print(action)
                obs, rewards, dones, info = env.step(action)
                env.render()


finally:
    env.close()
    print("-----Carla Environment is closed-----")
