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
from pygame.locals import K_ESCAPE
from gym.spaces import Box, Dict
from carla_rllib.wrappers.carla_wrapper import DiscreteWrapper
from carla_rllib.wrappers.carla_wrapper import ContinuousWrapper
from matplotlib import pyplot as plt
import cv2

class BaseEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array', 'state_pixels']}

    def __init__(self, config):

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
        self._obs_shape = (80,80,1)
        self._cum_reward = 0

        # Declare remaining variables
        self.frame = None
        self.timeout = 2.0

        # Memory for reward functions
        self.velocity = 0
        self.lane_invasion = 0

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
        spawn_points = self.world.get_map().get_spawn_points()[
            :self._num_agents]
        if self._agent_type == "continuous":
            for n in range(self._num_agents):
                self._agents.append(ContinuousWrapper(self.world,
                                                      spawn_points[n],
                                                      self._render_enabled))
        elif self._agent_type == "discrete":
            for n in range(self._num_agents):
                self._agents.append(DiscreteWrapper(self.world,
                                                    spawn_points[n],
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
        if  (self.frame - self.start_frame) == 1:
            print("Epsiode reward: ", self._cum_reward)
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

        # Extract observations for agents
        obs_dict = dict()
        for agent in self._agents:
            obs_dict[agent.id] = agent.state.image
        
        obs_dict["Agent_1"] = cv2.resize(obs_dict["Agent_1"], (self._obs_shape[0],self._obs_shape[1]))
        obs_dict["Agent_1"] = cv2.cvtColor(obs_dict["Agent_1"], cv2.COLOR_RGB2GRAY)

        # PLOTTING - Be careful, this is slow!
        # plt.ion()
        # plt.show()
        # plt.imshow(obs_dict["Agent_1"], cmap="gray")
        # plt.draw()
        # plt.pause(1e-6)

        obs_dict["Agent_1"] = obs_dict["Agent_1"].reshape(obs_dict["Agent_1"].shape[0],obs_dict["Agent_1"].shape[1],1)
        return obs_dict

    def _calculate_reward(self, agent):
        """Return the current reward"""
        # Get agent sensor measurements
        lane_invasion = agent.state.lane_invasion
        velocity = agent.state.velocity
        collisions  = agent.state.collision 
        
        # Calculate temporal differences
        invasions_incr = lane_invasion - self.lane_invasion
        collision_penalty = 0
        if collisions == True:
            collision_penalty = 100

        # Update values
        self.lane_invasion = lane_invasion

        reward = -0.1
        reward = reward + velocity - 50 * int(invasions_incr) - collision_penalty
        return reward

    def _is_done(self):
        """Return the current terminal condition"""
        done_dict = dict()
        for agent in self._agents:
            done_dict[agent.id] = agent.state.terminal
        return done_dict

    def _get_info(self):
        """Return current information"""
        # TODO: add something to print out
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
            reset = dict(
                Agent_1=dict(position=(46.1, 208.9),
                             yaw=0,
                             steer=0,
                             acceleration=-1.0),
                Agent_2=dict(position=(56.1, 208.9),
                             yaw=0,
                             steer=0,
                             acceleration=-1.0)
            )
        else:
            reset = dict(
                Agent_1=dict(position=(0, 0),
                             yaw=0,
                             velocity=(1, 0),
                             acceleration=(0, 0)),
                Agent_2=dict(position=(-10, 0),
                             yaw=0,
                             velocity=(1, 0),
                             acceleration=(0, 0))
            )
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
