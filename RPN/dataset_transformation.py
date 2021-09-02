from numpy.lib.arraysetops import ediff1d
from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import random
import os
import json
from collections import Counter

import utils_functions as utils
from config import ModelConfig
import tensorflow as tf

config = ModelConfig()
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

data_directory = "data/"
annotation_file_template = "{}/{}/annotation{}.json"

TRAIN_IMAGES_DIRECTORY = "data/train/images"
TRAIN_ANNOTATIONS_PATH = "data/train/annotations.json"

VAL_IMAGES_DIRECTORY = "data/val/images"
VAL_ANNOTATIONS_PATH = "data/val/annotations.json"

# we need to do the same thing both for train and val set
for path in [TRAIN_ANNOTATIONS_PATH, VAL_ANNOTATIONS_PATH]:

    # directly load the dataset
    dataset = json.load(open(path, 'r'))

    # load the only accepted classes, with new ids and names, to change existing annotations
    accepted_ids = config.ACCEPTED_CLASSES_IDS
    new_names = config.NEW_NAMES
    old_new_ids = utils.group_classes(accepted_ids, new_names)

    # if we deleted elements from the "dataset["categories"]" list WHILE iterating inside the for loop
    # some of the classes would be skipped, so we save which to delete and we do it afterwards
    to_delete = []
    for category in dataset["categories"]:
        # for each category we gather the right dictionary
        my_dict = list(filter(lambda cat: cat['old_id'] == category["id"], old_new_ids))
        # if it is an accepted class, we will find the right dictionary
        if not my_dict == []:
            # and we will change the id and names
            category["id"] = my_dict[0]["new_id"]
            category["name"] = my_dict[0]["new_name"]
            category["name_readable"] = my_dict[0]["new_name"]
        else:
            # otherwise we will delete the category
            to_delete.append(category)
    
    # here we delete all the non accepted categories
    for category in to_delete:
        dataset["categories"].remove(category)
    
    # being a list of dictionaries, it's possible to have duplicates
    # so we transform each dict into tuples, then we use "set" to exclude duplicates
    seen = set()
    new_l = []
    for d in dataset["categories"]:
        t = tuple(d.items())
        if t not in seen:
            seen.add(t)
            new_l.append(d)
    dataset["categories"] = new_l

    # exactly the same with annotations, but there is only the id that needs to be changed
    to_delete = []
    for annotation in dataset["annotations"]:
        my_dict = list(filter(lambda cat: cat['old_id'] == annotation["category_id"], old_new_ids))
        if not my_dict == []:
            # and we will change the id and names
            annotation["category_id"] = my_dict[0]["new_id"]
        else:
            # otherwise we will delete the category
            to_delete.append(annotation)

    # not really sure we need to delete them, maybe better for speed
    for annotation in to_delete:
        dataset["annotations"].remove(annotation)

    # UNCOMMENT WHEN READY
    # with open(path, "w") as outfile:
    #     json.dump(dataset, outfile)

