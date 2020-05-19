

import tensorflow as tf
import keras
from keras import layers
import numpy as np


"""
The model class is a simple model built using Keras' functional API.
This isn't necessary, we can also use Keras' Sequential model
It has one hidden layer with 10 hidden units.
"""
class model(tf.keras.Model):
    def __init__(self, num_states, num_actions, weights):
        super(model, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.myWeights = weights #These are the weights, calculated beforehand

        """
        In the layer whose weights you want to initialize, set the kernel_initializer argument equal to an object of
        the initialize_weights class (below). Pass the corresponding weights to the constructor
        """
        self.layer1 = layers.Dense(10, kernel_initializer=initialize_weights(self.myWeights))
        self.layer2 = layers.Dense(num_actions)

    #Feed forward function
    def call(self, inputs):
        x = self.layer1(inputs)
        out = self.layer2(x)

        return out


"""
By subclassing the Keras Initializer class, you can create your own custom weight initializers.
These subclasses need to have at least a constructor, which calls the Initializer constructor and defines your weights,
and a function __call__, which has the exact signature listed below
"""
class initialize_weights(keras.initializers.Initializer):
    def __init__(self, weights):
        super(initialize_weights, self).__init__()
        self.myWeights = weights

    def __call__(self, shape, dtype=None): #Shape is passed implicitly by layers.Dense in the model class. It returns the shape of the weights of the layer
        #You can write any code in here, but since we already have the weights calculated ahead of time, just return. This __call__ function is called
        #by line 22 above
        return self.myWeights



"""
Simple agent to run the feed-forward part of the model (no training is implemented in this example)
"""
class Agent:
    def __init__(self, num_states, num_actions, weights):
        self.num_states = num_states
        self.num_actions = num_actions
        self.myWeights = weights

        self.model = model(self.num_states, self.num_actions, self.myWeights)

    def run(self, inputs):
        output = self.model(inputs)
        print(output)
        return output


"""
You need to make sure the weights passed into the initialize_weights object are the correct shape.
The shape must be the same as that required by the layer
"""

#We start with manual weights, with a shape of (dimState x numUnits). Where dimState is the dimension of our observation (in this case 2),
#and numUnits is the number of hidden units in the layer whose weights we are initializing (in this case 10)

initWeights = [[.3, .1, .9, 2.0, 1.0, 3.1, .2, .6, .4, .9],
               [.2, .3, .4, .1, .6, .4, .8, .9, .2, .8]] # np.shape(initWeights) = (2, 10)

#The next three functions are used to make the weights compatible with Keras' dense layer
initWeights = np.asarray(initWeights, dtype='float32') #Create a numpy array
initWeights = np.atleast_2d(initWeights) #The numpy array must be at least rank 2 (Dense layer requirement)
initWeights = tf.convert_to_tensor(initWeights) #Converts numpy array to tensor

#We use the same operations on the observation to make it compatible with the input layer of our model
state = [2, 3]
state = np.asarray(state, dtype='float32')
state = np.atleast_2d(state)
state = tf.convert_to_tensor(state)

#If done correctly, no errors
agent = Agent(10, 1, initWeights)
agent.run(state)


