import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt


def myhiddenfunction(x):
    return (np.sin(x)**2 + np.cos(x)/3 + 1)/2


def generator(batchsize):
    while True:
        inputs = np.random.uniform(low=-np.pi, high=np.pi, size=batchsize)
        outputs = np.zeros(batchsize)
        for i in range(0, batchsize):
            outputs[i] = myhiddenfunction(inputs[i])
        yield inputs, outputs


input_layer = layers.Input(shape=(1,))
x = layers.Dense(20, activation="relu")(input_layer)
x = layers.Dense(30, activation="relu")(x)
x = layers.Dense(20, activation="relu")(x)
output_layer = layers.Dense(1, activation="relu")(x)

mymodel = tf.keras.models.Model(input_layer, output_layer)
mymodel.summary()
mymodel.compile(optimizer="adam", loss="mse")
batchsize = 64
mymodel.fit_generator(generator(batchsize), steps_per_epoch=1000, epochs=10)

x = np.arange(-np.pi, np.pi, 0.05)
y = [myhiddenfunction(a) for a in x]
z = [z[0] for z in mymodel.predict(np.array(x))]
plt.plot(x, y, 'r', x, z, 'b')
plt.show()

