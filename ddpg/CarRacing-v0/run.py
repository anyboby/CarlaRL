import gym
from ddpg import DDPG
from gym.envs.box2d.car_racing import CarRacing

env = CarRacing()
ddpg = DDPG((15,15,1), 3, 10)
ddpg.load_weights('_LR_0.0005_actor.h5', '_LR_0.0005_critic.h5')
ddpg.train(env, render=True, batch_size=64, nb_episodes=10000)
env.close()
