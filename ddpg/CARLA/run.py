import argparse
import os
import carla_rllib
import json
import gym
from carla_rllib.environments.carla_envs.config import BaseConfig, parse_json
from ddpg import DDPG


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
try:
    env = gym.make("CarlaBaseEnv-v0", config=configs[0])
    obs, reward, done, info = env.step([0,0])
    print(obs.shape)
    ddpg = DDPG((50, 50, 3), 2, 10)
    #ddpg.load_weights('_LR_0.0005_actor.h5', '_LR_0.0005_critic.h5')
    ddpg.train(env, render=False, batch_size=64, nb_episodes=10000)
finally:
    env.close()
    print("-----Carla Environment is closed-----")