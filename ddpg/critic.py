from keras.models import Model
from keras.initializers import RandomUniform
from keras.layers import Dense,  Conv2D, Flatten, Input, MaxPool2D, concatenate
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam 
import keras.backend as K



class Critic:
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
        self.model.compile(Adam(self.learning_rate), 'mse')
        # Target net for stability
        self.target_model = self._build_model()        
        self.target_model.compile(Adam(self.learning_rate), 'mse')
        # Function to compute Q-value gradients (Actor Optimization)
        self.action_grads = K.function([self.model.input[0], self.model.input[1]], K.gradients(self.model.output, [self.model.input[1]]))
    '''
    Build a convolutional neural net with 3 output neurons
    '''
    def _build_model(self):
        state = Input((self.state_size))
        x = Conv2D(2, kernel_size=4, activation='relu', input_shape=self.state_size)(state)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = Conv2D(4, kernel_size=4, activation='relu')(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = BatchNormalization()(x)
        
        # Actions
        action_shape = (self.action_size,)
        action_layer = Input(shape=action_shape)
        
        # TODO: In the original paper the actions are merged in the second hidden layer
        x = Flatten()(x)
        x = concatenate([Dense(128, activation='relu')(x), action_layer])
        x = Dense(128, activation='relu')(x)
        out = Dense(1, activation='linear', kernel_initializer=RandomUniform(minval=-0.05,maxval=0.05))(x)
        return Model([state, action_layer], out)
    
    def gradients(self, states, actions):
        """ Compute Q-value gradients w.r.t. states and policy-actions
        """
        return self.action_grads([states, actions])
    
    def target_predict(self, inp):
        """ Prediction of target network
        """
        return self.target_model.predict(inp)
    # Why does the Critic have no predict function
    
    def train_on_batch(self, states, actions, critic_target):
        """ Train the critic network on batch of sampled experience
            using the keras function train_on_batch
        """
        return self.model.train_on_batch([states, actions], critic_target)
    
    def transfer_weights(self):
        """ Transfer model weights to target model with a factor of Tau
        """
        W, target_W = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau)* target_W[i]
        self.target_model.set_weights(target_W)
    def save(self, path):
        self.model.save_weights(path + '_critic.h5')

    def load_weights(self, path):
        self.model.load_weights(path)
         
#critic = Critic((96,96,1), 3, 0.001, 0.1)
#critic.model.summary()
