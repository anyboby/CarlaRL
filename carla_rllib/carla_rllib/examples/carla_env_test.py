import argparse
import os
import carla_rllib
import json
import gym
import numpy as np
from carla_rllib.environments.carla_envs.config import BaseConfig, parse_json

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

try:
    env = gym.make("CarlaBaseEnv-v0", config=configs[0])

    obs = env.reset()

    y = 0
    t = 0
    print("-----Carla Environment is running-----")
    # import time
    while True:

        y += 1

        t += 0.15
        s = 0.3 * np.sin(t)
        a = 0.8
        # Discrete Actions
        # action = dict(Agent_1=[y, y, 0],
        #                    Agent_2=[y-6, y, 0])

        # Continuous Action
        action = dict(Agent_1=[s, a],
                      Agent_2=[s, a])

        obs, reward, done, info = env.step(action)
        print(env._agents["Agent_1"].state)
        # time.sleep(0.1)
        if any(d for d in done.values()):
            env.reset()
            y = 0
            t = 0

finally:
    env.close()
    print("-----Carla Environment is closed-----")
