from keras.models import Model
from keras.initializers import RandomUniform
from keras.layers import Dense, Conv2D, Flatten, Input, MaxPool2D, concatenate
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
import keras.backend as K
import numpy as np
import tensorflow as tf


class Actor:
    '''
    Set basic parameters for the model
    '''
    def __init__(self, state_size, action_size, learning_rate, tau):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.tau = tau
        # Actual model
        self.model = self._build_model()
        # Target net
        self.target_model = self._build_model()
        self.adam_optimizer = self.optimizer()
    '''
    Build a convolutional neural net with 3 output neurons
    '''
    def _build_model(self):
        
        state = Input((self.state_size))
        # Convolutions
        x = Conv2D(2, kernel_size=4, activation='relu', input_shape=self.state_size)(state)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = Conv2D(4, kernel_size=4, activation='relu') (x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        
        # Connect convolution and dense layers
        # 2D -> 1D (Linearization)
        x = Flatten()(x)
        
        # 3 hidden layers
        x = Dense(300, activation='relu')(x)
        # Creates 512 x 512 weights
        x = Dense(600, activation='relu')(x)
        x = BatchNormalization()(x)
        
        # Defining the output for each dimension seperately
        steering = Dense(1,activation='tanh',kernel_initializer=RandomUniform(minval=-0.05,maxval=0.05))(x)   
        acceleration = Dense(1,activation='sigmoid',kernel_initializer=RandomUniform(minval=-0.05,maxval=0.05))(x)   
        brake = Dense(1,activation='sigmoid',kernel_initializer=RandomUniform(minval=-0.05,maxval=0.05))(x) 
        out = concatenate([steering,acceleration,brake],axis=-1)
        
        model = Model(input=state,output=out)        
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
    def predict(self, state):
        """ Prediction of actor network
        """        
        action = self.model.predict(np.expand_dims(state, axis=0))
        # Normalize the steering between -1 and 1
        # Only used if sigmoid function
        # action[0] = (action[0] * 2) - 1; 
        return action
    def target_predict(self, inp):
        """ Prediction of target network
        """
        return self.target_model.predict(inp)
    def transfer_weights(self):
        """ Transfer model weights to target model with a factor of Tau
        """
        W, target_W = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau)* target_W[i]
        self.target_model.set_weights(target_W)
    def train(self, states, actions, grads):
        """ Actor Training
        """
        self.adam_optimizer([states, grads])
    def optimizer(self):
        """ Actor Optimizer
        """
        action_gdts = K.placeholder(shape=(None, self.action_size))
        params_grad = tf.gradients(self.model.output, self.model.trainable_weights, -action_gdts)
        grads = zip(params_grad, self.model.trainable_weights)
        return K.function([self.model.input, action_gdts], [tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)][1:])
    def save(self, path):
        self.model.save_weights(path + '_actor.h5')

    def load_weights(self, path):
        self.model.load_weights(path)
#actor = Actor((20,20,1), 3, 0.001, 0.1)
#actor.model.summary()
