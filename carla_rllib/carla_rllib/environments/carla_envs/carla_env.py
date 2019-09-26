"""CARLA Gym Environment
This script provides single- and multi-agent environments for Reinforcement Learning in the Carla Simulator.

Class:
    * BaseEnv - environment base class
    * SAEnv - environment with one agent
    * MAEnv - environment with multiple agent
"""
import sys
import os
import glob
import time
try:
    sys.path.append(glob.glob('%s/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        os.environ["CARLA_ROOT"],
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
import gym
import pygame
import numpy as np
import cv2
import random
from pygame.locals import K_ESCAPE
from gym.spaces import Box, Dict
from carla_rllib.wrappers.carla_wrapper import DiscreteWrapper
from carla_rllib.wrappers.carla_wrapper import ContinuousWrapper
from carla_rllib.wrappers.carla_wrapper import DataGeneratorWrapper
from carla_rllib.wrappers.carla_wrapper import BirdsEyeWrapper
from carla_rllib.wrappers.carla_wrapper import FrontAEWrapper
import carla_rllib.utils.reward_functions as rew_util

from matplotlib import pyplot as plt
import cv2
import random

class BaseEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array', 'state_pixels']}

    def __init__(self, config):

        # some flags for wrapper selection, only use one
        self._data_gen = False
        self._use_front_ae = False
        self._use_birdseye = False
        data_gen_shape, front_ae_shape, birdseye_shape = (64,64,1), (64,), (1,12,18,64) 

        print("-----Starting Environment-----")
        # Read config
        self._stable_baselines = config.stable_baselines
        self._agent_type = config.agent_type
        self._sync_mode = config.sync_mode
        self._delta_sec = config.delta_sec
        self._render_enabled = config.render
        self._host = config.host
        self._port = config.port
        self._map = config.map
        self._num_agents = config.num_agents
        self._obs_shape = (64,64,1)
        if self._use_front_ae:
            self._obs_shape = front_ae_shape
        elif self._use_birdseye:
            self._obs_shape = birdseye_shape
        elif self._data_gen:
            self._obs_shape = data_gen_shape


        self._cum_reward = 0
        self._prev_action = None
        self._action = None
        self._good_spawn_points = { 
            "Town05" : (51.1, 205.3)
        }
        self._current_position = None
        # Declare remaining variables
        self.frame = None
        self.timeout = 100.0

        # Memory for reward functions
        self.velocity = 0
        self.lane_invasion = 0
        self.dist_to_middle_lane = 0
        self.MAX_DIST_MIDDLE_LANE = 1.8
        self.MAX_VELOCITY = 40 # TODO: Find out
        self.MAX_SPEED_LIMIT = 10

        # Initialize client and get/load map        
        try:
            client = carla.Client(self._host, self._port)
            self.world = client.get_world()
            if (self._map and self.world.get_map().name != self._map):
                client.set_timeout(100.0)
                print('Load map: %r.' % self._map)
                self.world = client.load_world(self._map)
            client.set_timeout(2.0)
            print("Connected to Carla Server")
        except:
            raise ConnectionError("Cannot connect to Carla Server!")

        # Enable/Disable Synchronous Mode
        self._settings = self.world.get_settings()
        if self._sync_mode:
            if not self._settings.synchronous_mode:
                _ = self.world.apply_settings(carla.WorldSettings(
                    no_rendering_mode=False,
                    synchronous_mode=True,
                    fixed_delta_seconds=self._delta_sec))
            print("Synchronous Mode enabled")
        else:
            if self._settings.synchronous_mode:
                _ = self.world.apply_settings(carla.WorldSettings(
                    no_rendering_mode=False,
                    synchronous_mode=False,
                    fixed_delta_seconds=None))
            print("Synchronous Mode disabled")

        # Create Agent(s)
        self._agents = []
        self.spawn_points = self.world.get_map().get_spawn_points() #commented by @Moritz [:self._num_agents]
        if self._data_gen:
            # prefer out of town spawn spots
            for n in range(self._num_agents):
                self._agents.append(DataGeneratorWrapper(self.world,
                                                         self.spawnPointGenerator(self.spawn_points),
                                                         self._render_enabled))
        elif self._use_birdseye:
            for n in range(self._num_agents):
                self._agents.append(BirdsEyeWrapper(self.world,
                                                      self.spawn_points[random.randint(0,len(self.spawn_points))],
                                                      self._render_enabled))

        elif self._use_front_ae:
            for n in range(self._num_agents):
                self._agents.append(FrontAEWrapper(self.world,
                                                      self.spawn_points[random.randint(0,len(self.spawn_points))],
                                                      self._render_enabled))                                                
        
        elif self._agent_type == "continuous":
            # Good spawn points for training:
            for n in range(self._num_agents):
                self._agents.append(ContinuousWrapper(self.world,
                                                      self.spawn_points[random.randint(0,len(self.spawn_points))],
                                                      self._render_enabled))
                
        elif self._agent_type == "discrete":
            for n in range(self._num_agents):
                self._agents.append(DiscreteWrapper(self.world,
                                                    self.spawn_points[random.randint(0,len(self.spawn_points))],
                                                    self._render_enabled))


            else:
                raise ValueError(
                    "Agent type not available. Adjust config and choose one from: ['continuous', 'discrete']")




        # Baseline support
        self.action_space = None
        self.observation_space = None
        if (self._stable_baselines and
            self._agent_type == "continuous" and
                self._num_agents == 1):
            low = np.array([-1.0, -1.0])
            high = np.array([1.0, 1.0])
            self.action_space = Box(low, high, dtype=np.float32)
            self.observation_space = Box(low=0, high=255,
                                         shape=self._obs_shape,
                                         dtype=np.uint8)
            print("Baseline support enabled")
        else:
            print("Baseline support disabled\n" +
                  "(Note: Baselines are only supported for single agent with continuous control)")

        # Frame skipping
        if self._agent_type == "continuous":
            self._frame_skip = config.frame_skip
            print("Frame skipping enabled")
        else:
            self._frame_skip = 0
            print("Frame skipping disabled")

        # Spawn agents

        # Hacky workaround to solve waiting time when spawned:
        # Unreal Engine simulates starting the car and shifting gears,
        # so you are not able to apply controls for ~2s when an agent is spawned
        for agent in self._agents:
            agent._vehicle.apply_control(
                carla.VehicleControl(manual_gear_shift=True, gear=1))
        self.start_frame = self.world.tick()
        for agent in self._agents:
            agent._vehicle.apply_control(
                carla.VehicleControl(manual_gear_shift=False))
        if self._render_enabled:
            self.render()
        print("Agent(s) spawned")

    def step(self, action):
        """Run one timestep of the environment's dynamics

        Single-Agent Environment:
        Accept a list of actions and return a tuple (observations, reward, terminal and info)

        Multi-Agent Environment:
        Accept a list of actions stored in a dictionary for each agent and
        return dictionaries accordingly (observations, rewards, terminals and infos)

        Parameters:
        ----------
            action: list (dict)
                action(s) provided by the agent(s)

        Returns:
        ----------
            obs_dict: list (dict)
                observation(s) of the current environment
            reward_dict: float (dict)
                reward(s) returned after previous actions
            done_dict: bool (dict)
                whether the episode of the agent(s) is(are) done
            info_dict: dict
                contains auxiliary diagnostic information
        """
        # Set step and initialize reward_dict
        self._action = action
        reward_dict = dict()

        for agent in self._agents:
            if self._num_agents == 1:
                agent.step(action)
            else:
                agent.step(action[agent.id])
            reward_dict[agent.id] = 0

        # Run step, update state, calculate reward and check for dead agent
        done = 0
        for _ in range(self._frame_skip + 1):
            self.frame = self.world.tick()
            if self._render_enabled:
                self.render()
            for agent in self._agents:
                done += agent.update_state(self.frame,
                                           self.start_frame,
                                           self.timeout)
                reward_dict[agent.id] += self._calculate_reward(agent)
            if done:
                break

        
        self._cum_reward = self._cum_reward + reward_dict["Agent_1"]
        if  (self.frame - self.start_frame) == 1 + self._frame_skip:
            print("\033[94m ################ Epsiode reward: ", self._cum_reward, "############ \x1b[0m")
            self._cum_reward = 0
        # Retrieve observations, terminal and info
        obs_dict = self._get_obs()
        done_dict = self._is_done()
        info_dict = self._get_info()

        if self._num_agents == 1:
            return obs_dict["Agent_1"], reward_dict["Agent_1"], done_dict["Agent_1"], info_dict["Agent_1"]
        else:
            return obs_dict, reward_dict, done_dict, info_dict

    def preprocess(self, obs_arr):
        return obs_arr



    def reset(self):
        """Reset the state of the environment and return initial observations

        Returns:
        ---------- 
            obs_dict: dict
                the initial observations
        """

        # Set reset
        for agent in self._agents:
            reset = self._get_reset(agent)
            agent.reset(reset)

        # Run reset and update state
        self.start_frame = self.world.tick()
        if self._render_enabled:
            self.render()
        for agent in self._agents:
            agent.update_state(self.start_frame,
                               self.start_frame,
                               self.timeout)

        # Retrieve observations
        obs_dict = self._get_obs()

        if self._num_agents == 1:
            return obs_dict["Agent_1"]
        else:
            return obs_dict

    def render(self, mode='human'):
        """Render display"""
        for agent in self._agents:
            agent.render()

    def _get_obs(self):
        """Return current observations

        ---Note---
        Pull information out of state
        """
        if self._data_gen:
            obs_dict = dict()
            for agent in self._agents:
                obs_dict[agent.id] = agent.state.storage
                for sensor_id in agent.state.storage:
                    #print ("sensor id:" + str(sensor_id) + " image shape: " + str(agent.state.storage[sensor_id].shape))
                    # print ("image shape: " + str(agent.state.image.shape()))
                    # obs_dict[agent.id] = cv2.resize(obs_dict[agent.id], (self._obs_shape[0],self._obs_shape[1]))
                    # print("after resize: " + str(obs_dict[agent.id].shape()))
                    
                    a = 1
                    # cv2.imshow(sensor_id, obs_dict[agent.id][sensor_id])
                    # cv2.waitKey(1)

        elif self._use_birdseye:
            obs_dict = dict()
            for agent in self._agents:
                obs_dict[agent.id] = agent.state.image

            # PLOTTING - Be careful, this is slow!
            plot = False
            if (self.frame and plot and ((self.frame - self.start_frame) % 100) == 0):
                plt.ion()
                plt.show()
                plt.imshow(obs_dict["Agent_1"], cmap="gray")
                plt.draw()
                plt.pause(0.01)

            return obs_dict

        elif self._use_front_ae:
            obs_dict = dict()                                        
            for agent in self._agents:
                obs_dict[agent.id] = agent.state.image
            # cv2.imshow("test", obs_dict["Agent_1"])
            # cv2.waitKey(1)

            return obs_dict

        else:
            # Extract observations for agents
            obs_dict = dict()
            for agent in self._agents:
                obs_dict[agent.id] = agent.state.image
                #print ("image shape: " + str(agent.state.image.shape))
                obs_dict[agent.id] = cv2.resize(obs_dict[agent.id], (self._obs_shape[0],self._obs_shape[1]))
                #print("after resize: " + str(obs_dict[agent.id].shape))
            
            #obs_dict["Agent_1"] = cv2.cvtColor(obs_dict["Agent_1"], cv2.COLOR_RGB2GRAY)
            # cv2.imshow("image", obs_dict["Agent_1"])
            # cv2.waitKey(1)

            # PLOTTING - Be careful, this is slow!
            plot = False
            if (self.frame and plot and ((self.frame - self.start_frame) % 100) == 0):
                from PIL import Image
                im = Image.fromarray(obs_dict["Agent_1"] * 255)
                im.show()

                #plt.ion()
                #plt.show()
                #plt.imshow(obs_dict["Agent_1"], cmap="gray")
                #plt.draw()
                #plt.pause(0.01)

            obs_dict["Agent_1"] = obs_dict["Agent_1"].reshape(obs_dict["Agent_1"].shape[0],obs_dict["Agent_1"].shape[1],1)

        return obs_dict

    def _calculate_reward(self, agent):
        """Return the current reward"""
        # Get agent sensor measurements
        lane_invasion = agent.state.lane_invasion
        velocity = agent.state.velocity
        collisions  = agent.state.collision 
        dist_to_middle_lane = agent.state.distance_to_center_line
        current_steering = self._action[0]
        position = agent.state.position        
        delta_heading = agent.state.delta_heading

        # Calculate temporal differences and penalty values
        invasions_incr = lane_invasion - self.lane_invasion
        steering_change = 0
        position_change = 0
        #if not self._prev_action is None:
        steering_change = abs(current_steering) #penalty for steering at all -self._prev_action[0]
        if not self._current_position is None:
            position_change = abs(position[0] - self._current_position[0]) + abs(position[1] - self._current_position[1])
        # Hotfix because on collision this value is set negative
        if invasions_incr < 0:
            invasions_incr = 0
        dist_to_middle_lane_incr = self.dist_to_middle_lane - dist_to_middle_lane
        collision_penalty = 0
        if collisions == True:
            collision_penalty = 10
        # Update memory values
        self.lane_invasion = lane_invasion
        self.dist_to_middle_lane = dist_to_middle_lane
        self._prev_action = self._action
        self._current_position = position
        reward = -0.1
        reward = 0.1 * (reward + velocity * 0.2 - collision_penalty - (dist_to_middle_lane**2 - 0.4 * abs(delta_heading)))
        print("reward: " + str(reward))
        return reward

    def _is_done(self):
        """Return the current terminal condition"""
        done_dict = dict()

        for agent in self._agents:
            done_dict[agent.id] = agent.state.terminal
        return done_dict

    def _get_info(self):
        info_dict = dict()
        for agent in self._agents:
            info_dict[agent.id] = dict(Info="Store whatever you want")
        return info_dict

    def _get_reset(self, agent):
        """Return reset information for an agent

        ---Note---
        Implement your reset information here and
        adjust wrapper reset function if necessary
        """
        if self._agent_type == "continuous":
            reset = dict()
            for any_agent in self._agents:
                #@git from Moritz
                if self._data_gen:
                    position = (self.spawnPointGeneratorTown5().location.x,
                                self.spawnPointGeneratorTown5().location.y)
                # @git from Flo
                spawnpoint = self.spawnPointGeneratorScenarioRunner()
                position = spawnpoint[0]
                yaw = spawnpoint[1]
#                else:
#                    pos = any_agent._vehicle.get_location()
#                    position = (pos.x, pos.y)
                reset[any_agent.id]=dict(position=position,
                             yaw=yaw,
                             steer=0,
                             acceleration=-1.0)
            # commented by @Moritz 
            # reset = dict(
            #     Agent_1=dict(position=(pos.x, pos.y),
            #                  yaw=0,
            #                  steer=0,
            #                  acceleration=-1.0),
            #     Agent_2=dict(position=(56.1, 208.9),
            #                  yaw=0,
            #                  steer=0,
            #                  acceleration=-1.0)
            # )
        else:
            reset = dict()
            for any_agent in self._agents:
                if self._map=="Town05":
                    position = (56.1, 208.9)
                else:
                    pos = any_agent._vehicle.get_location()
                    position = (pos.x, pos.y)
                reset[any_agent.id]=dict(position=position,
                             yaw=0,
                             velocity=(1,0),
                             acceleration=(0,0))
            # commented by  @Moritz
            # reset = dict(
            #     Agent_1=dict(position=(0, 0),
            #                  yaw=0,
            #                  velocity=(1, 0),
            #                  acceleration=(0, 0)),
            #     Agent_2=dict(position=(-10, 0),
            #                  yaw=0,
            #                  velocity=(1, 0),
            #                  acceleration=(0, 0))
            # )
        return reset[agent.id]

    def set_phase(self):
        """Set up curriculum phase"""
        raise NotImplementedError

    def close(self):
        """Destroy agent and reset world settings"""
        for agent in self._agents:
            agent.destroy()
        self.world.apply_settings(self._settings)
        pygame.quit()

    """
    returns spawnpoint somewhere in outer part of city
    """
    def spawnPointGenerator(self, spawn_points):
        n_spawnpoints = len(spawn_points)
        spawn_ind = random.randint(0,n_spawnpoints-1)
        spawn_point = spawn_points[spawn_ind]
        # while (abs(spawn_points[spawn_ind].location.x) + abs(spawn_points[spawn_ind].location.y)) < 150:
        #     spawn_ind = random.randint(0,n_spawnpoints-1)
        #     spawn_point = spawn_points[spawn_ind]
        print("generated spawn_point: " + str(spawn_point) + ", numbeR: " + str(spawn_ind))
        return spawn_point

    """
    returns spawnpoint somewhere in outer part of city
    """
    def spawnPointGeneratorTown5(self):

        # for inspecting available spawnpoints
        # for i in range(len(self.spawn_points)):
        #     sp = self.spawn_points[i]
        #     print(str(sp.location.x) + "   " + str(sp.location.y) + "   " + str(i))
        #     if 45 < sp.location.x < 55 and 200 < sp.location.y < 210:
        #         print(i)
            

        good_spawns = [268, 212, 262, 227, 162, 50, 225, 97, 89, 50, 49, 163, 234, 235, 234, 234, 234, 235, 234, 235]
        n_spawnpoints = len(good_spawns)
        spawn_ind = good_spawns[random.randint(0,n_spawnpoints-1)]
        spawn_point = self.spawn_points[spawn_ind]
        print("generated spawn_point: " + str(spawn_point) + ", numbeR: " + str(spawn_ind))
        return spawn_point

    """
    returns spawn points of the scenarioRunner
    """
    def spawnPointGeneratorScenarioRunner(self):
        if (self._map == "Town05"):
            spawns = [[(47.24352264404297, -145.9805450439453), 2.5311660766601562], [(42.734130859375, 145.2400665283203), 1.4500408172607422]]
            spawn = random.choice(spawns)
            return spawn
        elif (self._map == "Town07"):
            spawns = [[(72.23163604736328, -7.422206878662109), 62.16304397583008], [(-15.64913558959961, -243.93333435058594), -169.06280517578125]]
            spawn = random.choice(spawns)
            return spawn
        else:
            raise Exception("Map not supported yet")
