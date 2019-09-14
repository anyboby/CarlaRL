import argparse
import os
import carla_rllib
import json
import gym
import numpy as np
from carla_rllib.environments.carla_envs.config import BaseConfig, parse_json
import pygame
from pygame.locals import K_UP
from pygame.locals import K_DOWN
from pygame.locals import K_LEFT
from pygame.locals import K_RIGHT
import time 

if __name__ == "__main__": 
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
    env = gym.make("CarlaBaseEnv-v0", config=configs[0])
    clock = pygame.time.Clock()


    track = "Town05"

    try:
        obs = env.reset()
        a = 0
        steer_cache  = 0
        print("-----Carla Environment is running-----")
        y = 0
        episode = 0
        episodes_successful = 106
        frame = 0
        frames_per_episode = 1000
        storage = {}
        
        for _id in obs.keys():
            storage[_id] = np.zeros((
                obs[_id].shape[0],
                obs[_id].shape[1],
                obs[_id].shape[2],
                frames_per_episode,
                )).astype("uint8")

        while True:
            # milliseconds = clock.get_time()
            # for event in pygame.event.get():
            #     keys = pygame.key.get_pressed()
            #     if keys[K_UP]:
            #         a = 1.0
            #     elif keys[K_DOWN]:
            #         a = -1.0
            #     else:
            #         a = 0     

            #     steer_increment = 9e-4 * milliseconds
            #     if  keys[K_LEFT]:   
            #         steer_cache -= steer_increment
            #     elif keys[K_RIGHT]:
            #         steer_cache += steer_increment
            #     else:
            #         steer_cache = 0.0
            
            # s = min(0.7, max(-0.7, steer_cache))
            
            # action = [s, a]
            # doesnt matter, autopilot anyways
            action = [0,0]
            obs, reward, done, info = env.step(action)

            for id_ in obs.keys():
                data = obs[id_]
                storage[id_][..., frame] = data
    
            #print(env._agents[0].state)

            frame += 1
            clock.tick(200) #@MORITZ TODO reset to original 4 (but seems to be laggy)
            #print(done)# = False
            y +=1
            if y%2 == 0:
                #print("reward",reward)
                y = 0
            if frame == frames_per_episode:
                print ("Success! 1000 frames reached! ending episode")
                for id_ in obs.keys():
                    np.save("camera_storage/{}_{}_{}.npy".format(id_, track, episodes_successful), storage[id_])
                    #pd.DataFrame(log_dicts).to_csv('logs/{}_{}.txt'.format(track, episode), index=False)
                episodes_successful+=1
                if episodes_successful == 110: break
                done = True
            if done:
                frame = 0
                episode +=1
                print ("episode {} done".format(episode))
                env.reset()

    finally:
        env.close()
        print("-----Carla Environment is closed-----")
