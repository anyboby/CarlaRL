"""
inspired by morvanzhou
"""

import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import gym
import os
import shutil
import matplotlib.pyplot as plt
import time

from master_network import MasterNetwork
from agent import Agent
import constants as Constants
import network_shares as Netshare

if __name__ == "__main__":
    Netshare.SESS = tf.Session()
    env = gym.make(Constants.GAME)

    # N_S[0] has None for sample placeholders, followed by the state space shape
    Netshare.DIM_S = [None]
    if Constants.manual_dims:
        Netshare.DIM_S.extend(Constants.DIMS_S)
    else:
        Netshare.DIM_S.extend(np.asarray(env.observation_space.shape))

    # N_A[0] has None for sample placeholders, followed by the action space shape
    Netshare.DIM_A = [None]
    if Constants.manual_dims:
        Netshare.DIM_A.extend(Constants.DIMS_A)
    else:    
        Netshare.DIM_A.extend(np.asarray(env.action_space.shape))
    
    print (env.action_space.shape)

    # Currently not manual Boundary option
    # A_BOUND[0] contains the minimums, A_BOUND[1] contains the maximums, each dependent on action space shape
    Netshare.BOUND_A = np.vstack((np.array(env.action_space.low), np.array(env.action_space.high)))

    print ("N_S: " + str(Netshare.DIM_S))
    print ("N_A: " + str(Netshare.DIM_A))
    print ("A_BOUND: " + str(Netshare.BOUND_A))

    with tf.device("/cpu:0"):
        #TODO die optimizer aufräumen
        Netshare.OPT_A = tf.train.RMSPropOptimizer(Constants.LR_A, name='RMSPropA')
        Netshare.OPT_C = tf.train.RMSPropOptimizer(Constants.LR_C, name='RMSPropC')
        GLOBAL_AC = MasterNetwork(Constants.GLOBAL_NET_SCOPE)  # we only need its params
        workers = []
        # Create worker
        for i in range(Constants.N_WORKERS-1):
            i_name = 'W_%i' % i   # worker name
            workers.append(Agent(i_name, GLOBAL_AC))
        # Create one worker with opencv showing 
        workers.append(Agent("W_%i"%Constants.N_WORKERS, GLOBAL_AC, cvshow=Constants.IMSHOW, render=Constants.RENDER))

    Netshare.SESS.run(tf.global_variables_initializer())

    if Constants.OUTPUT_GRAPH:
        if os.path.exists(Constants.LOG_DIR):
            shutil.rmtree(Constants.LOG_DIR)
        tf.summary.FileWriter(Constants.LOG_DIR, Netshare.SESS.graph)

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
        
    Netshare.COORD.join(worker_threads)

    plt.plot(np.arange(len(Constants.GLOBAL_RUNNING_R)), Constants.GLOBAL_RUNNING_R)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()
