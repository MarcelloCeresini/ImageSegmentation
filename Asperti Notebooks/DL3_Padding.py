import keras
from keras.datasets import cifar10
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras import metrics
import os
import numpy as np

x = layers.Input((32, 32, 3))
y = layers.Conv2D(15, kernel_size=3, strides=2, padding="same")(x)
model = Model(x, y)

model.summary()

# quit()

batch_size = 64
num_classes = 10
epochs = 100
