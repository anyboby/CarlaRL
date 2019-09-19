from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
#%load_ext autoreload
#%autoreload 2
from IPython.core.debugger import set_trace
import numpy as np
import time
import matplotlib.pyplot as plt
from utils import get_X_and_Y, get_data_gen, batcher, batcher_rgb, plot_semantic, make_movie


DECIMATION = 2
BATCH_SIZE = 10

CAMERA_IDS = [
    "FrontSS", "LeftSS", "RightSS", "RearSS", "FrontRGB", "LeftRGB", "RightRGB", "RearRGB", "TopSS"
]

INPUT_IDS = [
    "FrontRGB", "LeftRGB", "RightRGB", "RearRGB", "TopSS"
]


CLASSES_NAMES = [
    ['Roads', 'RoadLines'],
    
    ['Sidewalks'],
    
    ['Buildings'],
    
    ['Fences', 'Other', 'Pedestrians',
     'Poles', 'Walls', 'TrafficSigns'],
    
    ['Vehicles'],
    
    ['Vegetation'],
        
    ['None']
]


# CLASSES_NAMES = [
#     ['Roads', 'RoadLines'],
    
#     ['None', 'Buildings', 'Fences', 'Other', 'Pedestrians',
#      'Poles', 'Walls', 'TrafficSigns',
#      'Vegetation', 'Sidewalks'],
    
#     ['Vehicles'],
# ]
CLASSES_NAMES = [
    ['Roads', 'RoadLines'],
    
    ['None', 'Buildings', 'Fences', 'Other', 'Pedestrians',
     'Poles', 'Walls', 'TrafficSigns',
     'Vegetation', 'Sidewalks'],
    
    ['Vehicles'],
]

start_time = time.time()
storage = get_X_and_Y(['Town05'], range(1), DECIMATION, CAMERA_IDS)
X = [storage[id_] for id_ in CAMERA_IDS if "Top" not in id_]
Y = [storage[id_] for id_ in CAMERA_IDS if "Top" in id_][0]
print("\nReading data took {:.2f} [s]".format(time.time() - start_time))
# for camera_data in storage.values():
    # print(len(camera_data))

train_gen = batcher_rgb(
    get_data_gen(X, Y, classes_names=CLASSES_NAMES),
    BATCH_SIZE
)

valid_gen = batcher_rgb(
    get_data_gen(X, Y, flip_prob=0.0, validation=False, classes_names=CLASSES_NAMES),
    BATCH_SIZE
)

all_inputs, all_outputs = next(train_gen)
rgb_cams, top_ss, ss_cams = all_inputs[:4], all_inputs[4], all_outputs[:4]

input_shape = rgb_cams[0].shape[1:]
print(input_shape)


import pickle

from sklearn.metrics import roc_auc_score

import keras.backend as K
from keras.models import Model
from keras.layers import Add, Subtract, Average, Flatten, Reshape, Dense, Input, Lambda, Concatenate, Softmax, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, Convolution2DTranspose
from keras.optimizers import Adam, Nadam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from model import Deeplabv3

## --------------- allow dynamic memory growth to avoid cudnn init error ------------- ##
from keras.backend.tensorflow_backend import set_session #---------------------------- ##
import tensorflow as tf #------------------------------------------------------------- ##
config = tf.ConfigProto() #----------------------------------------------------------- ##
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU- ##
#config.log_device_placement = True  # to log device placement ----------------------- ##
sess = tf.Session(config=config) #---------------------------------------------------- ##
set_session(sess) # set this TensorFlow session as the default session for Keras ----- ##
## ----------------------------------------------------------------------------------- ##



def get_conv_encoder_model(
    input_and_output_shape,
    num_layers=4, central_exp=3,
    act='elu', l2_reg=1e-3
):
    x = inp = Input(input_and_output_shape)

    for i in range(num_layers, 0, -1):
        x = Convolution2D(2**(central_exp+i), (3, 3), activation=act, padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        
    return Model(inp, x, name='encoder_submodel')
    
    
def get_conv_decoder_model(
    encoded_shape, output_shape,
    num_layers=4, central_exp=3,
    act='elu', l2_reg=1e-3
):
    x = inp = Input(encoded_shape)
    
    num_channels = output_shape[-1]
    
    for i in range(1, num_layers+1):
        x = Convolution2D(2**(central_exp+i), (3, 3), activation=act, padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        
    x = Convolution2D(num_channels, (3, 3), activation='linear', padding='same')(x)

    return Model(inp, x, name='decoder_submodel')


def get_multi_model(
    input_shape, input_names,
    num_ae_layers=4, central_ae_exp=3,
    num_reconstruction_layers=3, central_reconstruction_exp=6,
    act='elu', l2_reg=1e-3,
):
    inputs = {
        inp_name: Input(input_shape, name='input_from_'+inp_name)
        for inp_name in input_names
    }

    image_shape = input_shape[:2]
    num_channels = input_shape[-1]
    
    encoder_model = get_conv_encoder_model(
        input_shape, num_ae_layers,
        central_ae_exp, act, l2_reg,
    )
    encoded_shape = K.int_shape(encoder_model.output)[1:]
    
    decoder_model = get_conv_decoder_model(
        encoded_shape, input_shape,
        num_ae_layers, central_ae_exp, act, l2_reg,
    )
    
    all_bottlenecks = {}
    ae_outputs = {}
    for inp_name in input_names:
        inp = inputs[inp_name]
        
        bttlnck = encoder_model(inp)
        all_bottlenecks[inp_name] = bttlnck

        decoded = decoder_model(bttlnck)
        ae_outputs[inp_name] = decoded
        ae_outputs[inp_name] = Softmax(axis=3, name='decoded_from_'+inp_name)(ae_outputs[inp_name])
    
    for_final_reconstruction = []
    side_cameras = [camera_id for camera_id in INPUT_IDS if 'Top' not in camera_id]
    for inp_name in side_cameras:
        x = Flatten()(all_bottlenecks[inp_name])
        for _ in range(num_reconstruction_layers-1):
            x = Dense(
                2**central_reconstruction_exp,
                activation=act,
                kernel_regularizer=l2(l2_reg),
            )(x)

        x = Dense(
                encoded_shape[0] * encoded_shape[1] * encoded_shape[2],
                activation=act,
                kernel_regularizer=l2(l2_reg),
            )(x)

        x = Reshape(encoded_shape)(x)
        x = Convolution2D(
            encoded_shape[-1], 
            (3, 3),
            activation=act,
            padding='same',
            name='encoded_from_'+inp_name,
        )(x)
        for_final_reconstruction.append(x)
        
    x = Concatenate()(for_final_reconstruction)
    encoded_reconstruction = Convolution2D(
        encoded_shape[-1],
        (3, 3),
        activation=act,
        padding='same',
        name='before_reconstruction',
    )(x)
                
    encoded_diff = Subtract(name='encoded_from_TopSS-encoded_reconstruction')([all_bottlenecks['TopSS'], encoded_reconstruction])

    reconstruction = decoder_model(encoded_reconstruction)
    reconstruction = Softmax(axis=3, name='reconstruction')(reconstruction)
    
    outputs = (
        [ae_outputs[inp_name] for inp_name in input_names]
        + [reconstruction]
        + [encoded_diff]
    )
    inputs = [inputs[inp_name] for inp_name in input_names]

    return Model(inputs, outputs)

num_ae_layers = 3
central_ae_exp = 5
patience = 10
num_sweeps = 24
validation_episodes_for_movies = [57, 101]

bottleneck_dim = (
    BATCH_SIZE,
    input_shape[0] // 2**num_ae_layers,
    input_shape[1] // 2**num_ae_layers,
    2**(central_ae_exp + 1),
)

print("bottleneck dim: " + str(bottleneck_dim))

zero_array = np.zeros(bottleneck_dim).astype('float32')

multi_model = get_multi_model(
    input_shape, INPUT_IDS,
    num_ae_layers, central_ae_exp,
    num_reconstruction_layers=3, central_reconstruction_exp=6,
    act='elu', l2_reg=1e-3,
)
multi_model.compile(
    loss=6*['categorical_crossentropy'] + ['mse'],
    loss_weights=5*[1] + [1] + [1],
    optimizer=Adam(1e-4)
)

early_stopping = EarlyStopping(
    monitor='val_reconstruction_loss',
    patience=patience,
    restore_best_weights=True,
)

multi_model.summary()

storage = get_X_and_Y(['Town05'], [108, 109], DECIMATION, CAMERA_IDS)
X_val = [storage[id_] for id_ in CAMERA_IDS if 'Top' not in id_]
Y_val = [storage[id_] for id_ in CAMERA_IDS if 'Top' in id_][0]
valid_gen = batcher_rgb(
    get_data_gen(X_val, Y_val, flip_prob=0.0, val_part=1, validation=True, classes_names=CLASSES_NAMES),
    BATCH_SIZE,
    zero_array,
)

MULTI_MODEL_EPISODES = [
    range(0, 8),
    range(8, 16),
    range(16, 24),
    range(24, 32),
    range(32, 40),
    range(40,48),
    range(48, 56),
    range(56, 64),
    range(72, 80),
    range(88, 96),
    range(96, 102),
]

# I've also tried our a recurrent model, for which I used
# a disjoint set of episodes (see the `recur_model.ipynb` for details)
RECURRENT_EPISODES = [
    
]


for sweep in range(num_sweeps):
    histories = []
    if sweep % 6 == 0:
        multi_model.optimizer = Adam(1e-4)
        
    for episodes in MULTI_MODEL_EPISODES:
        start_time = time.time()
        storage = get_X_and_Y(['Town05'], episodes, DECIMATION, CAMERA_IDS, storage)
        X = [storage[id_] for id_ in CAMERA_IDS if 'Top' not in id_]
        Y = [storage[id_] for id_ in CAMERA_IDS if 'Top' in id_][0]
        print('\nReading data took {:.2f} [s]'.format(time.time() - start_time))

        train_gen = batcher_rgb(
            get_data_gen(X, Y, val_part=10**10, classes_names=CLASSES_NAMES),
            BATCH_SIZE,
            zero_array,
        )
        
        all_inputs, all_outputs = next(train_gen)
        one_batch_X, one_batch_Y = all_inputs[:4], all_inputs[4]

        preds = multi_model.predict(all_inputs)
        preds = [pred[0:1] for pred in preds]

        for j, x in enumerate([x[0:1] for x in one_batch_X]):
            sep = np.zeros_like(x[:, :, ::5])
            #plot_semantic(np.concatenate([x, sep, preds[j]], axis=2))

        #plot_semantic(np.concatenate([one_batch_Y[0:1], sep, preds[4]], axis=2))
        #plot_semantic(np.concatenate([one_batch_Y[0:1], sep, preds[5]], axis=2))
        
        print('\n#### episodes = {} #### (sweep: {})'.format(episodes, sweep))

        history = multi_model.fit_generator(
            train_gen,
            steps_per_epoch=X[0].shape[-1] // BATCH_SIZE // 10,
            epochs=50,
            validation_data=valid_gen,
            validation_steps=X_val[0].shape[-1] // BATCH_SIZE // 2,
            verbose=1,
            callbacks=[early_stopping],
        )
        
        histories.append(history.history)
        
    val_loss = history.history['val_reconstruction_loss'][-(patience+1)]
    model_filename = 'models/multi_model__sweep={}_decimation={}_numclasses={}_valloss={:.3f}.h5'.format(sweep, DECIMATION, len(CLASSES_NAMES), val_loss)
    multi_model.save(model_filename)
    histories_filename = 'histories/multi_model__sweep={}_decimation={}_numclasses={}_valloss={:.3f}.pkl'.format(sweep, DECIMATION, len(CLASSES_NAMES), val_loss)
    with open(histories_filename, 'wb') as output:
        pickle.dump(histories, output)
        
    print('Metrics on one valid batch:')
    # x_y, _ = next(valid_gen)
    # one_batch_X, one_batch_Y = x_y[:4], x_y[4]
    all_inputs, all_outputs = next(train_gen)
    one_batch_X, one_batch_Y = all_inputs[:4], all_inputs[4]
    preds = multi_model.predict(all_inputs)
    for class_idx, class_names in enumerate(CLASSES_NAMES):
        print('\nClasses: {}'.format(class_names))
        # only the score for the birdseyeview reconstruction
        class_true = one_batch_Y[..., class_idx].flatten()
        class_pred = preds[5][..., class_idx].flatten()
        auc_score = roc_auc_score(class_true, class_pred)
        print('---> ROC AUC score: {:.1f}'.format(100*auc_score))
        print('---> class_pred.mean() / (class_true.mean() + 1e-10): {:.2f}'.format(class_pred.mean() / (1e-10 + class_true.mean())))

    for racetrack in ['Town05']:
        for episode in validation_episodes_for_movies:
            make_movie(
                model_filename,
                racetrack,
                episode,
                DECIMATION,
                CLASSES_NAMES,
                INPUT_IDS,
                multi_model,
                batch_size=BATCH_SIZE,
            )