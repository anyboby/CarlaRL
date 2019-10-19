from utils import make_movie
import numpy as np 
from keras import backend as K

DECIMATION = 2
BATCH_SIZE = 10

CAMERA_IDS = [
    'FrontSS', 'LeftSS', 'RightSS', 'RearSS', 'TopSS'
]

CAMERA_IDS_RGB = [
    "FrontRGB", "LeftRGB", "RightRGB", "RearRGB", "TopSS"
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

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    #you need this to load the model (weird, keras!)
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss



from keras.models import load_model

model_filename = "models/multi_model_rgb_sweep=18_decimation=2_numclasses=3_valloss=0.059.h5"
weights_loss = np.array([0.18,0.18,0.64])
weighted_ce_loss =  weighted_categorical_crossentropy(weights_loss)

multi_model = load_model(model_filename, custom_objects={"loss":weighted_ce_loss})
multi_model.summary()
racetrack = "Town05"
episodes = [13, 34, 45, 87, 90, 109] 

# for rgb use cmap = None and CAMERA_IDS_RGB
# for ss use cmap = gist_stern and CAMERA_IDS

# for semantic segmentation
# for episode in episodes:
#     make_movie(model_filename, racetrack, episode, DECIMATION, CLASSES_NAMES, CAMERA_IDS, episode_len=1000, batch_size=BATCH_SIZE)
#for rgb (cmap none)
for episode in episodes:
    make_movie(model_filename, racetrack, episode, DECIMATION, CLASSES_NAMES, CAMERA_IDS_RGB, multi_model=multi_model, episode_len=1000, batch_size=BATCH_SIZE, cmap=None)

