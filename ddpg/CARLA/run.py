import gym
import macad_gym
from ddpg import DDPG
env = gym.make("HomoNcomIndePOIntrxMASS3CTWN3-v0")

ddpg = DDPG((15,15,1), 3, 10)
#ddpg.load_weights('_LR_0.0005_actor.h5', '_LR_0.0005_critic.h5')
ddpg.train(env, render=True, batch_size=64, nb_episodes=10000)
env.close()
