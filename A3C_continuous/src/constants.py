import numpy as np
import tensorflow as tf
import multiprocessing
import threading

#Pendulum-v0
#CarRacing-v0
#CartPole-v1
#MountainCarContinuous-v0


GAME = "Pendulum-v0"
N_WORKERS = 4   #multiprocessing.cpu_count()
MAX_EP_STEP = 200
MAX_GLOBAL_EP = 2000
GLOBAL_NET_SCOPE = "Global_Net"
UPDATE_GLOBAL_ITER = 10  # very important param

TOGGLE_NSTEP = False 
GAMMA_N = 0.99
N_STEP_RETURN = 5
GAMMA_NN = GAMMA_N ** N_STEP_RETURN

GAMMA_V = 0.9
ENTROPY_BETA = 0.01
LR_A = 0.0001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0
TF_DEVICE = "/gpu:0"


EARLY_TERMINATION = 10000 # score difference between epMax and current score for termination
SKIP_STEPS = 0 #huge difference for mountaincar (try 1,2,3 a.s.o)

# network constants
# manual_dims is activated, state and action space can be manually set
# if deactivated, state and action space of env are used automatically as 
# network in/output
manual_dims = False
### CarRacing Dims ###
STATE_STACK = 4
STATE_WIDTH = 84
STATE_HEIGHT = 84
DIMS_S = [STATE_WIDTH, STATE_HEIGHT, STATE_STACK]
ACTIONS = 3
DIMS_A = [ACTIONS]
######################

#RENDERING AND OUTPUTS
OUTPUT_GRAPH = True
LOG_DIR = "./log"
LOG_DIR2 = "./log"
WAITKEY = 1
RENDER = False
IMSHOW = True
