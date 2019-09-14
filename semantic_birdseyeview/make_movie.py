from utils import make_movie

DECIMATION = 2
BATCH_SIZE = 10

CAMERA_IDS = [
    'FrontSS', 'LeftSS', 'RightSS', 'RearSS', 'TopSS'
]


CLASSES_NAMES = [
    ['Roads', 'RoadLines'],
    
    ['None', 'Buildings', 'Fences', 'Other', 'Pedestrians',
     'Poles', 'Walls', 'TrafficSigns',
     'Vegetation', 'Sidewalks'],
    
    ['Vehicles'],
]


## --------------- allow dynamic memory growth to avoid cudnn init error ------------- ##
from keras.backend.tensorflow_backend import set_session #---------------------------- ##
import tensorflow as tf #------------------------------------------------------------- ##
config = tf.ConfigProto() #----------------------------------------------------------- ##
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU- ##
#config.log_device_placement = True  # to log device placement ----------------------- ##
sess = tf.Session(config=config) #---------------------------------------------------- ##
set_session(sess) # set this TensorFlow session as the default session for Keras ----- ##
## ----------------------------------------------------------------------------------- ##

from keras.models import load_model

model_filename = "models/multi_model__sweep=10_decimation=2_numclasses=3_valloss=0.262.h5"
multi_model = load_model(model_filename)
multi_model.summary()
racetrack = "Town05"
episodes = [13, 34, 45, 87, 90, 109] 

for episode in episodes:
    make_movie(model_filename, racetrack, episode, DECIMATION, CLASSES_NAMES, CAMERA_IDS, episode_len=1000, batch_size=BATCH_SIZE, cmap="gist_stern")
