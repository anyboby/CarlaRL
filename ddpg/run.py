import gym
from ddpg import DDPG
from gym.envs.box2d.car_racing import CarRacing

env = CarRacing()
ddpg = DDPG((15,15,1), 3, 10)
#ddpg.load_weights('_LR_5e-05_actor.h5', '_LR_5e-05_actor.h5')
ddpg.train(env, render=False, batch_size=64, nb_episodes=1000)
env.close()
