"""
Usage: import the module or run from the command line as such:

    # Train a model from scratch
    python3 food.py train --dataset=/path/to/food_data/ --model=start

    # Continue training a model that you had trained earlier
    python3 food.py train --dataset=/path/to/food_data/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 food.py train --dataset=/path/to/food_data/ --model=last

    # Run food dataset evaluation on the last model you trained
    python3 food.py evaluate --dataset=/path/to/food_data/ --model=last
"""

import os
import sys
import time
import zipfile
import urllib.request
import shutil
import argparse

import numpy as np
from skimage import io, color, transform
import imgaug  # For image augmentation

from tensorflow import keras
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

from rpn_model import MaskRCNN
from config import ModelConfig
import utils_functions as utils

###############
###  PATHS  ###
###############

# Root directory of the project
ROOT_DIR = os.path.abspath("..") # TODO on final project, this will probably change.

# Path to trained weights file
FOOD_WEIGHTS_MODEL_PATH = os.path.join(ROOT_DIR, "rpn_weights", "rpn_weights_food.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


########################
###  CONFIGURATIONS  ###
########################

class InferenceConfig(ModelConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0


#################
###  DATASET  ###
#################

class FoodDataset():

    def __init__(self):
        # This list contains all image IDs
        self._image_ids = []
        # This list contains all image metadata
        self.image_info = [] # image width, height and annotations are saved here
        # Then we have a list to contain all classes information.
        # Background is always the first class
        self.class_info = [{
            "id": 0, 
            "name": "BG"
        }]

    def add_class(self, class_id, class_name):
        '''
        Adds a class ID to the list of classes of the dataset
        '''
        # Does the class exist already?
        if not class_id in {info['id'] for info in self.class_info}:
            # Only add the class if it's not already present
            self.class_info.append({
                "id": class_id,
                "name": class_name,
            })


    def add_image(self, image_id, path, **kwargs):
        '''
        Adds an image to the list of image IDs 
        '''
        image_info = {
            "id": image_id,
            "path": path,
        }
        image_info.update(kwargs) # image width, height and annotations are saved here
        self.image_info.append(image_info)


    def load_image(self, image_id):
        """
        Load the specified image and return it as a [H,W,3] numpy array.
        """
        # Load image
        image = io.imread(self.image_info[image_id]['path'])
        # Sometimes, the dataset has height and width of an image inverted.
        # Check this and rotate the image on the fly.
        if self.image_info[image_id]['height'] == image.shape[1] and \
            self.image_info[image_id]['width'] == image.shape[0] and \
            image.shape[0] != image.shape[1]:
            image = transform.rotate(image, 90, resize=True)
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image


    def load_food(self, dataset_dir, subset, 
                    class_ids=None,
                    return_coco=False):
        """
        Loads a subset of the dataset for the Food Recognition challenge (eg. train, val, val_subset)
        
        Inputs: 
        - dataset_dir: The root directory of the food dataset.
        - subset: What to load (train, val, val_subset)
        - class_ids: If provided, only loads images that have the given classes.
        - return_coco: If True, returns the COCO object.
        """
        general_path = os.path.join(dataset_dir, subset)
        image_dir = os.path.join(general_path, 'images')
        # Since the dataset adopts the COCO standard, we use the COCO 
        # class provided by pycocotools for managing image IDs, classes and
        # annotations.
        coco = COCO(os.path.join(general_path, 'annotations.json'))

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # Load all images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                # Only select images that contain one element of the class
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes by iterating on all class IDs
        for i in class_ids:
            self.add_class(i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                # The rest are args that denote the image's metadata.
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None))) # TODO: Are we dealing with crowds?
        if return_coco:
            return coco

        # Save the coco object in the class
        self._coco = coco
    

    def load_mask(self, image_id):
        """
        Load instance masks for the given image.

        This function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        instance_masks = []
        class_ids = []

        # Get image metadata and annotations from the provided ID.
        image_info = self.image_info[image_id]
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = annotation['category_id']
            if class_id:
                m = self._coco.annToMask(annotation)
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return self.create_empty_mask(image_id)


    def create_empty_mask(self, image_id):
        """
        Creates an empty mask in case none could be loaded for the image ID.
        """
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids


    def prepare(self):
        """
        Prepares the Dataset class for use.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Prepare additional properties for the dataset
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)


    @property
    def image_ids(self):
        return self._image_ids


############################################################
#  COCO Evaluation
############################################################

# def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
#     """
#     Arrange resutls to match COCO specs in http://cocodataset.org/#format
#     """
#     # If no results, return an empty list
#     if rois is None:
#         return []

#     results = []
#     for image_id in image_ids:
#         # Loop through detections
#         for i in range(rois.shape[0]):
#             class_id = class_ids[i]
#             score = scores[i]
#             bbox = np.around(rois[i], 1)
#             mask = masks[:, :, i]

#             result = {
#                 "image_id": image_id,
#                 "category_id": dataset.get_source_class_id(class_id, "coco"),
#                 "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
#                 "score": score,
#                 "segmentation": maskUtils.encode(np.asfortranarray(mask))
#             }
#             results.append(result)
#     return results


# def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
#     """Runs official COCO evaluation.
#     dataset: A Dataset object with valiadtion data
#     eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
#     limit: if not 0, it's the number of images to use for evaluation
#     """
#     # Pick COCO images from the dataset
#     image_ids = image_ids or dataset.image_ids

#     # Limit to a subset
#     if limit:
#         image_ids = image_ids[:limit]

#     # Get corresponding COCO image IDs.
#     coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

#     t_prediction = 0
#     t_start = time.time()

#     results = []
#     for i, image_id in enumerate(image_ids):
#         # Load image
#         image = dataset.load_image(image_id)

#         # Run detection
#         t = time.time()
#         r = model.detect([image], verbose=0)[0]
#         t_prediction += (time.time() - t)

#         # Convert model_dirresults to COCO format
#         # Cast masks to uint8 because COCO tools errors out on bool
#         image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
#                                            r["rois"], r["class_ids"],
#                                            r["scores"],
#                                            r["masks"].astype(np.uint8))
#         results.extend(image_results)

#     # Load results. This modifies results with additional attributes.
#     coco_results = coco.loadRes(results)

#     # Evaluate
#     cocoEval = COCOeval(coco, coco_results, eval_type)
#     cocoEval.params.imgIds = coco_image_ids
#     cocoEval.evaluate()
#     cocoEval.accumulate()
#     cocoEval.summarize()

#     print("Prediction time: {}. Average {}/image".format(
#         t_prediction, t_prediction / len(image_ids)))
#     print("Total time: ", time.time() - t_start)


##################
###  TRAINING  ###
##################

def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train RPN on Food Dataset.')
    parser.add_argument("command",
                        metavar="<command>",
                        choices=['train', 'evaluate'],
                        help="'train' or 'evaluate' on Food Recognition dataset")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/food/",
                        help='Directory of the Food Recognition dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = ModelConfig()
    else:
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        rpn = MaskRCNN(mode="training", config=config,
                                  out_dir=args.logs)
    else:
        rpn = MaskRCNN(mode="inference", config=config,
                                  out_dir=args.logs)

    # Select weights file to load
    # TODO: implement training code in new model
    if args.model.lower() == "last":
        # Find last trained weights
        model_path = rpn.find_last()
    elif args.model.lower() == "start":
        # Start from scratch (backbone is always trained on ImageNet)
        model_path = None
    else:
        model_path = args.model

    if model_path:
        # Load weights into the model held within the RPN class
        print("Loading weights ", model_path)
        rpn.model.load_weights(model_path)
        # Update the log dir
        rpn.set_log_dir(model_path)

    # Train or evaluate
    if args.command == "train":
        # Training dataset
        dataset_train = FoodDataset()
        dataset_train.load_food(args.dataset, "train")
        dataset_train.prepare()

        # Validation dataset
        dataset_val = FoodDataset()
        dataset_val.load_food(args.dataset, "val")
        dataset_val.prepare()

        # Image Augmentation
        # Right/Left flip 50% of the time
        augmentation = imgaug.augmenters.Fliplr(0.5)
        # TODO we could add other kinds of image augmentations here

        # TODO: define a better training schedule
        # Add a custom callback that reduces the learning rate during training
        # after epoch 50
        def scheduler(epoch, lr):
            if epoch < 50:
                return lr
            else:
                # Divide lr by 10
                return lr / 10

        custom_callbacks = [
            keras.callbacks.LearningRateScheduler(scheduler)
        ]

        # Fine tune all layers
        print("Fine tune all layers")
        rpn.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=60,                              # Start soft with 60 epochs
                    layers='heads',                         # training only the heads
                    augmentation=augmentation,
                    custom_callbacks=custom_callbacks)      # Add a custom callback that reduces the learning
                                                            # rate after some training steps.

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = FoodDataset()
        coco = dataset_val.load_food(args.dataset, 'val', return_coco=True)
        dataset_val.prepare()

        raise NotImplementedError

        # print("Running COCO evaluation on {} images.".format(args.limit))
        # evaluate_coco(morpndel, dataset_val, coco, "bbox", limit=int(args.limit))        

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))

