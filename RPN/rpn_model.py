import math
import numpy as np
import re
import datetime
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle

import tensorflow as tf
from tensorflow import keras
# Refer to TF's internal version of Keras for more stability
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
import tensorflow.keras.backend as K
import tensorflow.keras.losses as KLS

from skimage.transform import resize

class RPN():
    """
    This class encapsulates the RPN model and some of its functionalities.
    The Keras model is contained in the model property.
    """
    def __init__(self, mode, config, out_dir):
        """
        Inputs: 
        - mode: One of ["training", "inference"]
        - config: The model configuration object (ModelConfig object)
        - out_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.out_dir = out_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # TODO update names
            # \path\to\logs\food20211029T2315\rpn_food_0001.h5 (Windows)
            # /path/to/logs/food20211029T2315/rpn_food_0001.h5 (Linux)
            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]rpn\_food\_[\w-]+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                # Epoch number in file is 1-based, and in Keras code it's 0-based.
                # So, adjust for that then increment by one to start from the next epoch
                epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % epoch)

        # Directory for training logs
        log_dir = os.path.join(self.out_dir, "{}{:%Y%m%dT%H%M}".format(
            'food', now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        checkpoint_path = os.path.join(log_dir, "rpn_food_*epoch*.h5")
        checkpoint_path = checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")

        return log_dir, checkpoint_path