import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import Model

# Size of the input
input_dim = 784
# Size of the encoded representation
encoding_dim = 16
# Size of the middle layer
mid_dim = 64

# Input placeholder
input_img = layers.Input(shape=(input_dim, ))
# Definition of the network:
# The encoding part is just a simple network with two
# hidden layers: the first is a compression in a 64-dimensional
# space and the following is a further compression with 16 neurons
encoded = layers.Dense(mid_dim, activation='relu')(input_img)
encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
# The decoding part takes the 16-d encoded image and 
# brings it back to its 784-dimensional shape
decoded = layers.Dense(encoding_dim, activation='relu')(encoded)
decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)