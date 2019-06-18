import gym
from ddpg import DDPG

env = gym.make('CarRacing-v0')
ddpg = DDPG((96,96,1), 3, 10)
#ddpg.load_weights('_LR_5e-05_actor.h5', '_LR_5e-05_actor.h5')
ddpg.train(env, render=True, batch_size=32, nb_episodes=1000)
env.close()