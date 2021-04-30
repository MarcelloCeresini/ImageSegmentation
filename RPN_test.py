import os
import random
import datetime
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.models as KM
import keras.engine as KM
from tensorflow.python.ops.gen_data_flow_ops import stage

### INPUT IMAGE RELATED CONSTANTS ###
# Input image is resized using the "square" resizing mode from 
# the original implementation. 
# In this mode, images are scaled up such that the small side 
# is = IMAGE_MIN_DIM, but ensuring that the scaling doesn't 
# make the long side > IMAGE_MAX_DIM. Then the image is padded 
# with zeros to make it a square so multiple images can be put
# in one batch.
IMAGE_MIN_DIM = 800
IMAGE_MAX_DIM = 1024
# Image mean (RGB)
MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
# Define the image shape (1024x1024x3 if no changes to parameters are made)
IMAGE_SHAPE = np.array([IMAGE_MAX_DIM, IMAGE_MAX_DIM, 3])

### BACKBONE RELATED CONSTANTS ###
# Whether to use Resnet50 or Resnet101
BACKBONE_NETWORK = 'resnet50' # or 'resnet101'

### FPN RELATED CONSTANTS ###
# Size of the top-down layers used to build the feature pyramid
TOP_DOWN_PYRAMID_SIZE = 256

### RPN RELATED CONSTANTS ###
# Length of square anchor side in pixels
RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
# Ratios of anchors at each cell (width/height)
# A value of 1 represents a square anchor, and 0.5 is a wide anchor
RPN_ANCHOR_RATIOS = [0.5, 1, 2]
# Anchor stride
# If 1 then anchors are created for each cell in the backbone feature map.
# If 2, then anchors are created for every other cell, and so on.
RPN_ANCHOR_STRIDE = 1
# Non-max suppression threshold to filter RPN proposals.
# You can increase this during training to generate more propOsals.
RPN_NMS_THRESHOLD = 0.7
# How many anchors per image to use for RPN training
RPN_TRAIN_ANCHORS_PER_IMAGE = 256

# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True, train_bn=False):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    # See conv_block for comments on these operations.
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = KL.BatchNormalization(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = KL.BatchNormalization(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = KL.BatchNormalization(name=bn_name_base + '2c')(x, training=train_bn)

    # Input is added directly to the output. This is only possible if no
    # changes to the shapes are made during convolutions. This is true because
    # all convolutions are either (1,1) with unitarian strides or have same padding
    # (the one in the middle)
    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, train_bn=False):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 1x1 convolution with given stride + batch normalization and relu activation
    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = KL.BatchNormalization(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    # ...followed by a convolution with given filter, unitarian stride and same padding
    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = KL.BatchNormalization(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    # followed by a 1x1 convolution with unitarian stride that augments/reduces the channels
    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                  '2c', use_bias=use_bias)(x)
    x = KL.BatchNormalization(name=bn_name_base + '2c')(x, training=train_bn)

    # Also, a shortcut is added to link the input to the last of the three convolutions.
    # - Apply the chosen strides to compensate for skipping the first convolution
    # - No other changes are done, so just apply unitarian stride and use the same number
    #   of filters used for the third convolution
    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = KL.BatchNormalization(name=bn_name_base + '1')(shortcut, training=train_bn)

    # Adds together two tensors with the same shape.
    # In this case, it's an element wise sum between the output of the third convolution
    # and the input tensor.
    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x

def resnet_graph(input_image, stage5=False, train_bn=True):
    """Build a ResNet graph.
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layers

       Returns:
        [C1,C2,C3,C4,C5]: convolutional layers that can be shared 
            with other modules of the network
    """
    # Stage 1
    # Apply 3 rows and columns of zero padding.
    # 1024x1024x3 --> 1030x1030x3
    x = KL.ZeroPadding2D((3,3))(input_image)
    # Apply 7x7 convolution.
    # 1030x1030x3 --> ((1030-7+2*0)/2)+1x((1030-7+2*0)/2)+1x3
    # --> 512x512x64
    x = KL.Conv2D(filters=64, kernel_size=(7,7),
                    strides=(2,2), padding='valid',
                    name='conv1', use_bias=True)(x)
    # Apply Batch Normalization. It is said to have problems with
    # small batches during training, thus training is set to False
    x = KL.BatchNormalization(name='bn_conv1')(x, training=False)
    # Apply relu activation function
    x = KL.Activation('relu')(x)
    # 512x512x64 --> 256x256x64 because maxpooling with stride 2,
    # but using same padding.
    C1 = x = KL.MaxPooling2D(pool_size=(3,3), strides=(2,2),
                            padding='same')(x)
    # Stage 2
    # We use coarser/repeated convolution blocks defined in another function
    # 256x256x64 --> 256x256x256
    x = conv_block(input_tensor=x, kernel_size=3,
                filters=[64,64,256], stage=2,
                block='a', strides=(1,1))
    # 256x256x256 --> 256x256x256
    x = identity_block(x, 3, [64,64,256], stage=2, block='b')
    C2 = x = identity_block(x, 3, [64,64,256], stage=2, block='c')
    # Stage 3
    # 256x256x256 --> 128x128x512 (due to the stride (2,2) 
    x = conv_block(x, 3, [128,128,512], stage=3, block='a')
    x = identity_block(x, 3, [128,128,512], stage=3, block='b')
    x = identity_block(x, 3, [128,128,512], stage=3, block='c')
    C3 = x = identity_block(x, 3, [128,128,512], stage=3, block='d')
    # Stage 4
    # 128x128x256 --> 64x64x1024 
    # of the first convolutional block)
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    # resnet50 has 5 identity blocks, resnet101 has 22 identity blocks
    block_count = {'resnet50': 5, 'resnet101': 22}[BACKBONE_NETWORK]
    for i in range(block_count):
        x = identity_block(x, 3, [256,256,1024], stage=4, block=chr(98+i))
    C4 = x
    # Stage 5
    # 64x64x256 --> 32x32x2048
    if stage5:
        x = conv_block(x, 3, [512,512,2048], stage=5, block='a')
        x = identity_block(x, 3, [512,512,2048], stage=3, block='b')
        C5 = x = identity_block(x, 3, [512,512,2048], stage=3, block='c')
    else:
        C5 = None
    # Return the 5 shared convolutional layers
    return [C1,C2,C3,C4,C5]

def build():
    """
    Builds the Backbone + RPN model.
    """
    h, w = IMAGE_SHAPE[:2]
    if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
        raise Exception("Image size must be dividable by 2 at least 6 times "
                        "to avoid fractions when downscaling and upscaling."
                        "For example, use 256, 320, 384, 448, 512, ... etc. ")
    
    # Define inputs
    # a. The input image
    input_image = KL.Input(
        shape = IMAGE_SHAPE, name='input_image'
    )
    # b. The anchors in normalized coordinates
    # TODO: why?
    input_anchors = KL.Input(
        shape = [None, 4], name='input_anchors'
    )

    # Backbone: Bottom-up ResNet50 + Top-down FPN with
    # shared convolutional layers. This requires a custom implementation.

    ### BOTTOM-UP RESNET50 ###
    # Recall that:
    # C1 = 256x256x64
    # C2 = 256x256x256
    # C3 = 128x128x512
    # C4 = 64x64x1024
    # C5 = 32x32x2048
    C1,C2,C3,C4,C5 = resnet_graph(input_image, stage5=True, train_bn=False)

    ### TOP-DOWN FPN ###
    P5 = KL.Conv2D(TOP_DOWN_PYRAMID_SIZE, (1,1), name='fpn_c5p5')(C5)
    # P5 has shape 32x32x256
    P4 = KL.Add(name='fpn_p4add')([
        # UpSampling2D repeats rows and columns of the data (P5) 2 times.
        # Thus, this is 64x64x256
        KL.UpSampling2D(size=(2,2), name="fpn_p5upsampled")(P5),
        # C4 is transformed into 64x64x256
        KL.Conv2D(TOP_DOWN_PYRAMID_SIZE, (1,1), name='fpn_c4p4')(C4)
        # Hence the shapes match and we can perform an addition
    ])
    # P4 has shape 64x64x256
    P3 = KL.Add(name='fpn_p3add')([
        # 128x128x256
        KL.UpSampling2D(size=(2,2), name="fpn_p4upsampled")(P4),
        # 128x128x256
        KL.Conv2D(TOP_DOWN_PYRAMID_SIZE, (1,1), name="fpn_c3p3")(C3)
    ])
    # P3 has shape 128x128x256
    P2 = KL.Add(name='fpn_p2add')([
        # 256x256x256
        KL.UpSampling2D(size=(2,2), name="fpn_p3upsampled")(P3),
        # 256x256x256
        KL.Conv2D(TOP_DOWN_PYRAMID_SIZE, (1,1), name="fpn_c2p2")(C2)
    ])
    # P2 has shape 256x256x256
    # Attach 3x3 conv to all P layers to get the final feature maps.
    # All dimensions are kept the same
    P2 = KL.Conv2D(TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="same", name="fpn_p2")(P2)
    P3 = KL.Conv2D(TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="same", name="fpn_p3")(P3)
    P4 = KL.Conv2D(TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="same", name="fpn_p4")(P4)
    P5 = KL.Conv2D(TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="same", name="fpn_p5")(P5)
    # An additional feature map is generated by subsampling from P5
    # with stride 2
    P6 = KL.MaxPooling2D(pool_size=(1,1), strides=2, name='fpn_p6')(P5)
    # P6 has shape 16x16x256
    
    # List of feature maps for the rpn
    rpn_feature_maps = [P2, P3, P4, P5, P6]

    # In testing mode, anchors are given as input
    anchors = input_anchors

    # TODO




if __name__ == "__main__":
    model = build()