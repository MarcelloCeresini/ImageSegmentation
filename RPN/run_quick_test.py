import tensorflow as tf
import numpy as np
import os
import argparse

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle

from rpn_model import MaskRCNN
from food import FoodDataset
from config import ModelConfig
from data_generator import DataGenerator

import utils_functions as utils

argp = argparse.ArgumentParser()
argp.add_argument('--test_datagen', 
                    action='store_true', 
                    default=False, 
                    help='Use this flag to test the DataGenerator')
args = argp.parse_args()

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
model = MaskRCNN('inference', config)

# Create the DataLoader to test it.
if args.test_datagen and os.path.exists(os.path.join('..','data')):
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
    img = [mpimg.imread('../res/006626.jpg'), mpimg.imread('../res/007675.jpg')]

# The output of the detect function is a list of dictionaries like the following:
# {
#    "rpn_boxes": [M, (y1, x1, y2, x2)],
#    "rpn_classes": [M],
#    "rois": [N, (y1, x1, y2, x2)],
#    "class_ids": [N],
#    "scores": [N],
#    "masks": [H, W, N]
# }
results = model.detect(img)

RPN_BBOXES_TO_DRAW = 20

# Show each image sequentially and draw a selection of "the best" RPN bounding boxes.
for i in range(len(results)):
    image = img[i] # Detections are given with respect to the original image size
    rpn_classes = results[i]['rpn_classes']
    rpn_bboxes = results[i]['rpn_boxes']
    rois = results[i]['rois']
    classes = results[i]['class_ids']
    scores = results[i]['scores']
    masks = results[i]['masks']
    # Select positive rpn_bboxes
    condition_rpn = np.where(rpn_classes > 0.5)[0]
    # If there is at least a positive bbox, draw it, otherwise draw random ones
    if len(condition_rpn):
        rpn_bboxes = rpn_bboxes[condition_rpn]
    # Sort by probability
    rnd_rpn_bboxes = sorted(np.arange(0, rpn_bboxes.shape[0], 1),
                        key=lambda x, c=rpn_classes: c[x])[:RPN_BBOXES_TO_DRAW]
    rnd_rpn_bboxes = rpn_bboxes[rnd_rpn_bboxes, :]
    fig, ax = plt.subplots()
    # Note that if we used the DataGenerator, the image could have been normalized previously
    if args.test_datagen:
        ax.imshow(utils.denormalize_image(image, config.MEAN_PIXEL))
    else:
        ax.imshow(image)
    for bb in rnd_rpn_bboxes:
        rect = Rectangle(
            (bb[1], bb[0]), bb[3] - bb[1], bb[2] - bb[0],
            linewidth=1, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
    for bb in rois:
        rect = Rectangle(
            (bb[1], bb[0]), bb[3] - bb[1], bb[2] - bb[0],
            linewidth=1, edgecolor='g', facecolor='none'
        )
        ax.add_patch(rect)
    # TODO I HAVE NO IDEA HOW TO DRAW MASKS LOL
    #for m in range(masks.shape[-1]):
    #    ax.add_patch(m)
    plt.show()