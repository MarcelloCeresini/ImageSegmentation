import pylab
import json
import os

import utils_functions as utils
from config import ModelConfig

config = ModelConfig()
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

data_directory = "data"
train_dir = "train_original"
val_dir = "val_original"
train_out_dir = "train"
val_out_dir = "val"
annotations_filename = "annotations.json"

# Create directories for output paths just in case
os.makedirs(os.path.join(data_directory, train_out_dir), exist_ok=True)
os.makedirs(os.path.join(data_directory, val_out_dir), exist_ok=True)

TRAIN_IMAGES_DIRECTORY = os.path.join(data_directory, train_dir, "images")
TRAIN_ANNOTATIONS_PATH = os.path.join(data_directory, train_dir, annotations_filename)

VAL_IMAGES_DIRECTORY = os.path.join(data_directory, val_dir, "images")
VAL_ANNOTATIONS_PATH = os.path.join(data_directory, val_dir, annotations_filename)

TRAIN_OUT_ANNOTATIONS_PATH =  os.path.join(data_directory, train_out_dir, annotations_filename)
VAL_OUT_ANNOTATIONS_PATH =  os.path.join(data_directory, val_out_dir, annotations_filename)

# we need to do the same thing both for train and val set
for path_type, path in [("train", TRAIN_ANNOTATIONS_PATH), ("val", VAL_ANNOTATIONS_PATH)]:

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
    total_annotations_num = len(dataset["annotations"])
    for annotation in dataset["annotations"]:
        my_dict = list(filter(lambda cat: cat['old_id'] == annotation["category_id"], old_new_ids))
        if not my_dict == []:
            # and we will change the id and names
            annotation["category_id"] = my_dict[0]["new_id"]
        else:
            # otherwise we will delete the category
            to_delete.append(annotation)

    # remove these annotations from the dataset
    for annotation in to_delete:
        dataset["annotations"].remove(annotation)

    deleted_annotations_num = len(to_delete)
    print("In dataset of type {}, {} annotations have been deleted, so there are {}/{} annotations left".format(
        path_type, deleted_annotations_num, total_annotations_num - deleted_annotations_num, total_annotations_num
    ))

    # Save the new annotations file
    if path_type == "train":
        out_path = TRAIN_OUT_ANNOTATIONS_PATH
    elif path_type == 'val':
        out_path = VAL_OUT_ANNOTATIONS_PATH
    with open(out_path, "w") as outfile:
        json.dump(dataset, outfile)
