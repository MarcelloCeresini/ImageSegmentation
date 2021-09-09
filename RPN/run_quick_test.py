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
mod_images, rpn_classes_batch, rpn_bboxes_batch, detections_batch = model.detect(img)

# Detections from mrcnn are (y1, x1, y2, x2, class_id, score)

print("Shape of rpn_classes: {}".format(tf.shape(rpn_classes_batch)))
print("Shape of rpn_bboxes: {}".format(tf.shape(rpn_bboxes_batch)))
print("Shape of detections: {}".format(tf.shape(detections_batch)))

RPN_BBOXES_TO_DRAW = 20
MRCNN_BBOXES_TO_DRAW = 5

# Show each image sequentially and draw a selection of "the best" RPN bounding boxes.
# Note that the model is not trained yet so "the best" boxes are really just random.
for i in range(len(mod_images)):
    image = mod_images[i, :, :]
    rpn_classes = rpn_classes_batch[i, :]
    rpn_bboxes = rpn_bboxes_batch[i, :, :]
    bboxes = detections_batch[i, :, :4]
    classes = detections_batch[i, :, 4]
    scores = detections_batch[i, :, 5]
    # Select positive bboxes
    condition_rpn = np.where(rpn_classes > 0.5)[0]
    condition_mrcnn = np.where(scores > 0.5)[0]
    # If there is at least a positive bbox, draw it, otherwise draw random ones
    if len(condition_rpn):
        rpn_bboxes = rpn_bboxes[condition_rpn]
    if len(condition_mrcnn):
        bboxes = bboxes[condition_mrcnn]
    # Sort by probability
    rnd_rpn_bboxes = sorted(np.arange(0, rpn_bboxes.shape[0], 1),
                        key=lambda x, c=rpn_classes: c[x])[:RPN_BBOXES_TO_DRAW]
    rnd_mrcnn_bboxes = sorted(np.arange(0, bboxes.shape[0], 1),
                            key=lambda x, s=scores: s[x])[:MRCNN_BBOXES_TO_DRAW]
    rnd_rpn_bboxes = rpn_bboxes[rnd_rpn_bboxes, :]
    rnd_mrcnn_bboxes = bboxes[rnd_mrcnn_bboxes, :]
    rnd_rpn_bboxes = utils.denorm_boxes(rnd_rpn_bboxes, image.shape[:2])
    rnd_mrcnn_bboxes = utils.denorm_boxes(rnd_mrcnn_bboxes, image.shape[:2])
    fig, ax = plt.subplots()
    # Note that the image was previously normalized so colors will be weird
    ax.imshow(utils.denormalize_image(image, config.MEAN_PIXEL))
    for bb in rnd_rpn_bboxes:
        rect = Rectangle(
            (bb[1], bb[0]), bb[3] - bb[1], bb[2] - bb[0],
            linewidth=1, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
    for bb in rnd_mrcnn_bboxes:
        rect = Rectangle(
            (bb[1], bb[0]), bb[3] - bb[1], bb[2] - bb[0],
            linewidth=1, edgecolor='g', facecolor='none'
        )
        ax.add_patch(rect)
    plt.show()