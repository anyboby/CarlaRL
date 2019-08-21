"""Reward functions

This script provides examples of reward functions for reinforcement learning.

Functions:
    * reward_1 - (tbd)
    * reward_2 - (tbd)
    ...
"""
import numpy as np


def reward_1(distance_to_center_line, delta_heading, current_speed, target_speed,
             max_reward=1.0, min_reward=-1.0, weights=[0.6, 0.4, 0.5]):
    """Returns the reward based on distance to center line and delta heading"""
    # Distance reward 1
    distance_reward = max(
        max_reward - (distance_to_center_line / 0.6)**3, min_reward)

    # Distance reward 2 (DQN)
    # distance_reward = max(2.0 - (distance_to_center_line + 1.0, 8)**2, -2.0)

    # Distance reward 3
    # a maximal reward
    # b penalty clipping factor
    # c zeros
    # max(-(a / c**2) * (distance_to_center_line / 0.7 * 3.5)**2 + a, b);

    # Heading reward
    heading_reward = max(1 - delta_heading / (1/3 * 90), min_reward)

    # # Speed reward
    # if current_speed <= target_speed:
    #     speed_reward = max(-1, 1 + 10 * np.log(current_speed / target_speed))
    # else:
    #     speed_reward = max(2.0 - (current_speed / target_speed)**4, -1)

    # Combination
    reward = weights[0] * distance_reward + \
        weights[1] * heading_reward
    # weights[2] * speed_reward

    return reward


def reward_2(state):
    return 0
