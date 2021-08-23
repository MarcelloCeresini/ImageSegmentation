import tensorflow as tf
import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle

from rpn_model import RPN
from food import FoodDataset
from config import ModelConfig
from data_generator import DataGenerator

import utils_functions as utils

# GPU debugging
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth and only one visible GPU (for now)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

config = ModelConfig()
model = RPN('inference', config)

print(model.summary())

# Create the DataLoader to test it.
if os.path.exists(os.path.join('..','data')):
    # If we have the folder for the actual dataset (data, in the root folder), use it.
    dataset_val = FoodDataset()
    dataset_val.load_food('../data', 'val_subset')
    dataset_val.prepare()
    dataset_generator = DataGenerator(dataset_val, config, shuffle=True, 
                            dont_normalize=True) # The network already does its preprocessing
    data_iterator = dataset_generator.iterator
    batch = next(data_iterator)
    img = batch[0][0] # Images are already normalized because they are in validation format.
else:
    # Test the detection with two images in the res folder
    img = [mpimg.imread('res/006626.jpg'), mpimg.imread('res/007675.jpg')]
mod_images, rpn_classes, rpn_bboxes = model.detect(img)

print("Shape of rpn_classes: {}".format(tf.shape(rpn_classes)))
print("Shape of rpn_bboxes: {}".format(tf.shape(rpn_bboxes)))

BBOXES_TO_DRAW = 100

# Show each image sequentially and draw a selection of "the best" RPN bounding boxes.
# Note that the model is not trained yet so "the best" boxes are really just random.
for i in range(len(mod_images)):
    image = mod_images[i, :, :]
    classes = rpn_classes[i, :]
    bboxes = rpn_bboxes[i, :, :]
    # Select positive bboxes
    condition = np.where(classes > 0.5)[0]
    # If there is at least a positive bbox, draw it, otherwise draw random ones
    if len(condition):
        bboxes = bboxes[condition]
    # Sort by probability
    rnd_bboxes = sorted(np.arange(0, bboxes.shape[0], 1),
                        key=lambda x, c=classes: c[x])[:BBOXES_TO_DRAW]
    rnd_bboxes = bboxes[rnd_bboxes, :]
    rnd_bboxes = utils.denorm_boxes(rnd_bboxes, image.shape[:2])
    fig, ax = plt.subplots()
    # Note that the image was previously normalized so colors will be weird
    ax.imshow(utils.denormalize_image(image, config.MEAN_PIXEL))
    for bb in rnd_bboxes:
        rect = Rectangle(
            (bb[1], bb[0]), bb[3] - bb[1], bb[2] - bb[0],
            linewidth=1, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
    plt.show()