from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import decode_predictions, preprocess_input
import keras.backend as K

# Allows us to use K.gradients()
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import time
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the model and pretrained weights
model = VGG16(weights='imagenet', include_top=True)
model.summary()

# Example of classification
img_path = os.path.join('res','elephant.jpg')
img = image.load_img(img_path, target_size=(224,224))
# x0 will be a numpy array
x0 = image.img_to_array(img)
# Add dimension for batch
x = np.expand_dims(x0, axis = 0)

print("shape = {}, range=[{},{}]".format(x.shape,np.min(x[0]),np.max(x[0])))

preds = model.predict(x)
print("label = {}".format(np.argmax(preds)))
print("Predicted: ", decode_predictions(preds, top=3)[0])

# Create another copy of the original image
xd = image.array_to_img(x[0])
# Show the image
imageplot = plt.imshow(xd)
plt.show()

# FOOL THE NETWORK >:)

# The image we have fed the network
input_img = model.input

# build a loss function that maximizes the activation of a different category
pred = model.output

# Fix the category we want to maximize
output_index = 3 # tiger shark
# Construct the one-hot encoded expected classification vector
expected = np.zeros(1000)
expected[output_index] = 1
expected = K.variable(expected)

# We calculate the loss between our output and the fake output we wanted
loss = K.categorical_crossentropy(model.output[0], expected)

# compute the gradient of the input picture wrt this loss
grads = K.gradients(loss, input_img)[0]

# normalize the gradient
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

# define a function that returns the loss and grads given the input picture
loss_grads = K.function([input_img], [loss, grads])

input_img_data = np.copy(x)

# Run gradient ascent for 50 steps in order to minimize this loss
# This will make the prediction gradually shift to our desired selection
for i in range(100):
    print("iteration n. {}".format(i))
    res = model.predict(input_img_data)[0]
    print("elephant prediction: {}".format(res[386]))
    print("tiger shark prediction: {}".format(res[3]))
    time.sleep(0.1)
    loss_value, grads_value = loss_grads([input_img_data])
    ming = np.min(grads_value)
    maxg = np.max(grads_value)
    scale = 1/(maxg-ming)
    # Slightly modify the input image according to this gradient
    input_img_data -= grads_value * scale

# At this point, we can check that the prediction for this new image
# is our selected one.
preds = model.predict(input_img_data)
print("label = {}".format(np.argmax(preds)))
print('Predicted:', decode_predictions(preds, top=3)[0])

# Plot the original and the changed images side-by-side
img = input_img_data[0]
img = image.array_to_img(img)
plt.figure(figsize=(10,5))

ax = plt.subplot(1, 3, 1)
plt.title("elephant")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.imshow(xd)

ax = plt.subplot(1, 3, 2)
plt.imshow(img)
plt.title("tiger shark")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

imageplot = plt.imshow(img)
plt.show()



