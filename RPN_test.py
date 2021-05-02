import os
import random
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import numpy as np
import tensorflow as tf
import keras.layers as KL
import keras.models as KM
import keras.engine as KM

from skimage.transform import resize

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

# Strides used for computing the shape of each stage of the backbone network
# (when based on resnet50/101).
# This is used for aligning proposals to the image. If the image is 1024x1024,
# the 4 means that the first feature map P2 will be 1024/4 = 256x256. In the 
# second one we divide by 8 and so on. The last feature map (P6) is 1024/64=16.
# With these ratio indications we can easily express the relationship between a 
# feature map and the original image.
BACKBONE_STRIDES = [4,8,16,32,64]

### FPN RELATED CONSTANTS ###
# Size of the top-down layers used to build the feature pyramid
TOP_DOWN_PYRAMID_SIZE = 256

### RPN RELATED CONSTANTS ###
# Length of square anchor side in pixels
# You can see it as an area of the anchor in pixels if the anchor was squared.
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

################
### BACKBONE ###
################

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
        x = identity_block(x, 3, [512,512,2048], stage=5, block='b')
        C5 = x = identity_block(x, 3, [512,512,2048], stage=5, block='c')
    else:
        C5 = None
    # Return the 5 shared convolutional layers
    return [C1,C2,C3,C4,C5]

###########
### RPN ###
###########

def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    '''Builds the actual computation graph of the RPN.

    feature_map: backbone features [batch, height, width, depth]
    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).

    Returns:
        rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.

    '''
    # Make the feature map deeper
    # The result is the convolutional layer on which the RPN will evaluate anchors
    shared = KL.Conv2D(512, (3,3), padding='same', activation='relu',
                        strides=anchor_stride,
                        name='rpn_conv_shared')(feature_map)
    
    # This convolutional layer stores the anchor scores. As you can see, there are
    # double the expected anchors per location, because for each anchor we have
    # a foreground and a background score. For example, if anchors per location is 3,
    # We would have 6 scores per each pixel.
    # It's just a 1x1 convolution because we only need to create scores without touching the convolution
    # Also, we are not applying softmax yet because we might want to see the logits
    # Padding is valid but it's a 1x1 convolution so that doesn't really mean anything
    x = KL.Conv2D(2*anchors_per_location, (1,1), padding='valid', activation='linear',
                   name='rpn_class_raw')(shared)

    # Reshape the scores to [batch, anchors, 2]
    # Note that the -1 means that the number of anchors is inferred from the other dimensions
    rpn_class_logits = KL.Lambda(
        lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2])
    )(x)

    # Softmax on the logits to get probabilities FG/BG
    rpn_probs = KL.Activation(
        "softmax", name="rpn_class_xxx"
    )(rpn_class_logits)

    # Apply a bounding box refinement
    # The output of this layer will be a [batch, H, W, anchors per location * 4] tensor
    # meaning that for each pixel of the previous feature map (H,W) we will have the anchors
    # we wanted (anchors_per_location), each described by 4 numbers.
    # These 4 numbers are actually:
    # x,y: the refined center of the anchor
    # log(w), log(h): the refined width and height of the anchor
    x = KL.Conv2D(anchors_per_location*4, (1,1), padding='valid',
                    activation='linear', name='rpn_bbox_pred')(shared)
    
    ## As done before, we reshape this output to [batch, anchors, 4]
    rpn_bbox = KL.Lambda(
        lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4])
    )(x)

    # Return the obtained tensors
    return [rpn_class_logits, rpn_probs, rpn_bbox]


def build_rpn_model(anchor_stride, anchors_per_location, depth):
    '''Builds a Keras model for the RPN.

    anchors_per_location: the number of anchors per pixel in the feature map.
        Usually this number corresponds with the number of possible ratios of
        anchors
    anchor_stride: the stride to apply to the anchors generation. 1 means: generate
        one anchor per pixel in the feature map, while 2 means one anchor
        every 2 pixels.
    depth: depth of the backbone feature map

    Returns a Keras Model, which itself outputs:

    (Remember that each proposal is classified in one of two classes, namely
    foreground and background)
    rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
    rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
    rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                applied to anchors.
    '''
    input_feature_map = KL.Input(shape=[None,None, depth],
                            name='input_rpn_feature_map')
    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
    return KM.Model([input_feature_map], outputs, name='rpn_model')

##########################
### NETWORK DEFINITION ###
##########################

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

    # In testing mode, anchors are given
    anchors = input_anchors

    ### RPN MODEL ###
    # The RPN is a lightweight neural network that scans the image 
    # in a sliding-window fashion and finds areas that contain objects.
    # The regions that the RPN scans over are called anchors. 
    # Which are boxes distributed over the image area
    rpn_model = build_rpn_model(anchor_stride=RPN_ANCHOR_STRIDE,
                        anchors_per_location=len(RPN_ANCHOR_RATIOS), 
                        depth=TOP_DOWN_PYRAMID_SIZE)

    # We apply the RPN on all layers of the pyramid:
    layers_outputs = []
    for p in rpn_feature_maps:
        layers_outputs.append(rpn_model([p]))
    # Then we concatenate layer outputs this way:
    # Instead of dealing with the outputs of each layer separated from all others,
    # we want to deal with the outputs related to the same pixels of every layer.
    # For example, layers_outputs currently contains [[A1,B1,C1], [A2,B2,C2]] where
    # A,B,C are logits, probabilities and bboxes of layer 1 in the first list, layer 2 in
    # the second, etc. Instead, we want a list like [[A1,A2],[B1,B2],[C1,C2]] where
    # we relate the same things in different feature maps.
    outputs = list(zip(*layers_outputs))    
    # The asterisk makes zip "unzip" the list of lists by grouping the first, second, third... elements
    # Thus, outputs is exactly the list we wanted
    # Now, we want to concatenate the list of lists 
    output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
    # Finally, we concatenate all elements of the list as rows of three long tensors,
    # containing all logits, all class probabilities and all bounding boxes.
    outputs = [KL.Concatenate(axis=1, name=n)(list(o))
                for o,n in zip(outputs, output_names)]

    # Finally extract all tensors 
    rpn_class_logits, rpn_class, rpn_bbox = outputs

    model = KM.Model([input_image, input_anchors],
                     [rpn_class, rpn_bbox],
                     name='rpn')
    
    return model

###############################
### PREPROCESSING UTILITIES ###
###############################

def resize_image(image, min_dim, max_dim):
    '''
    Resizes an image by keeping the aspect ratio unchanged and using zero-padding
    to reshape it to a square.

    min_dim: the size of the smaller dimension
    max_dim: ensures that the image's longest side doesn't exceed this value

    Returns:
        image: the resized image
        window: (y1,x1,y2,x2): since padding might be inserted in the returned image,
            this window contains the coordinates of the image part in the full image.
            x2, y2 are not included, so the last "acceptable" line x2-1 and the last
            "acceptable" column is y2-1
        scale: the scale factor used to resize the image
        padding: type of padding added to the image [(top, bottom), (left, right), (0, 0)]
    '''
    # Keep track of the image dtype to return the same dtype
    image_dtype = image.dtype
    h, w = image.shape[:2]
    image_max = max(h, w)
    # Scale up, not down
    scale = max(1, min_dim/min(h,w))
    # Check if enlarging the max dim by scale would exceed our limit
    if round(image_max * scale) > max_dim:
        # If that's the case, bound the scale to the ratio between max dim and the image max
        scale = max_dim / image_max
    # Resize image using bilinear interpolation
    if scale != 1:
        image = resize(image, output_shape=(round(h * scale), round(w * scale)),
                        order=1, mode='constant', cval=0, clip=True,
                        anti_aliasing=False, anti_aliasing_sigma=False,
                        preserve_range=True)
    # Get new height and width:
    h, w = image.shape[:2]
    top_pad = (max_dim - h) // 2
    bottom_pad = max_dim - h - top_pad
    left_pad = (max_dim - w) // 2
    right_pad = max_dim - w - left_pad
    padding = [(top_pad, bottom_pad),
                (left_pad, right_pad),
                (0,0)]
    # Apply padding to the image
    image = np.pad(image, padding, mode='constant', constant_values=0)
    window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image.astype(image_dtype), window, scale, padding

##########################
### DETECTION PIPELINE ###
##########################

def preprocess_inputs(images):
    '''
    Takes a list of images, modifies them to the format expected as an
    input to the neural network.

    images: a list of image matrices with different sizes. What is constant
        is the third dimension of the image matrix, the depth (usually 3)
    
    Returns a numpy matrix containing the preprocessed image ([N, h, w, 3]).
    The preprocessing includes resizing, zero-padding and normalization.
    '''
    preprocessed_inputs = []
    windows = []
    for image in images:
        preprocessed_image, window, scale, padding = resize_image(
            image, IMAGE_MIN_DIM, IMAGE_MAX_DIM
        )
        # We want a normalized image, so we subtract the mean pixel to it
        # and convert to float.
        preprocessed_image = preprocessed_image.astype(np.float32) - MEAN_PIXEL
        preprocessed_inputs.append(preprocessed_image)
        windows.append(window)
    # Pack into arrays
    preprocessed_inputs = np.stack(preprocessed_inputs)
    windows = np.stack(windows)
    return preprocessed_inputs, windows

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors. It corresponds with the shape of one of the
            feature maps in the FPN (P2,P3,P4,P5,P6)
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all possible combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Ratios are width/height, scales are squared sizes of the sides of the anchors
    # if the anchor was a square.
    # Calculating the square root of the ratios we get the "average" unitary side of the anchor
    # For example, sqrt(0.5), sqrt(1) and sqrt(2)
    # If the square root is greater than 1, the width of the anchor box is greater than
    # its height and the opposite is also true. 
    # So, if we multiply scales for the square-rooted ratios, we get anchor widths
    # (for example, with ratio 2, the square root is 1.4... and the width will therefore be longer
    # than the one indicated in the "scales" array)
    # With the same reasoning we can say that dividing the scales by the squre-rooted ratios we get
    # heights.
    # Thus, if we want the heights we need to compute:
    heights = scales / np.sqrt(ratios)
    widths  = scales * np.sqrt(ratios)

    # What are the positions in the feature space?
    # We use arange to create evenly spaced sequences from 0 for all the
    # rows skipping the number of rows required by the anchor stride.
    # We multiply by the feature stride so that we get image-aligned coordinates
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    # Same for columns
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    # Generate all combinations
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of centers, widths and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # TODO: almost done!!!


def get_anchors(image_shape):
    """Returns the anchor pyramid for the given image size"""
    backbone_shapes = np.array([
        [int(math.ceil(image_shape[0] / stride)),
         int(math.ceil(image_shape[1] / stride))]
         for stride in BACKBONE_STRIDES]
    )
    anchors = []
    # The next function generates anchors at the different levels of the FPN.
    # Each scale is bound to a different level of the pyramid
    # On the other hand, all ratios of the proposals are used in all levels.
    for i in range(len(RPN_ANCHOR_SCALES)):
        anchors.append(generate_anchors(RPN_ANCHOR_SCALES[i],   # Use only the appropriate scale for the level
                                        RPN_ANCHOR_RATIOS,      # But use all ratios for the BBs
                                        backbone_shapes[i],     # At this level, the image has this shape...
                                        BACKBONE_STRIDES[i],    # Or better, is scaled of this quantity
                                        RPN_ANCHOR_STRIDE       # Frequency of sampling in the feature map 
                                                                # for the generation of anchors
                                        ))
    # Transform the list in an array [N, (y1,x1,y2,x2)] which contains all generated anchors
    # The sorting of the scale is bound to the scaling of the feature maps,
    # So first we have all anchors at scale 1/4, then all anchors at scale 1/8 and so on...
    anchors = np.concatenate(anchors, axis=0)

def detect(images):
    '''
    This function runs the detection pipeline.

    images: a list of images, even of different sizes 
        (they will be reshaped as zero-padded squares of the same dimensions)

    TODO: what does this function return?
    '''
    preprocessed_images, windows = preprocess_inputs(images)

    # Check that all images in the batch now have the same size
    image_shape = preprocessed_images[0].shape
    for g in preprocessed_images[1:]:
        assert g.shape == image_shape, \
            "All images must have the same size after preprocessing"
    
    # The network also receives anchors as inputs, so we need a function
    # that returns the anchors
    anchors = get_anchors(image_shape)



if __name__ == "__main__":
    # Instantiates the Mask-RCNN model up until the RPN
    # (with no post-processing)
    # (yet)
    model = build()

    # Test the detection with one image (stack it to simulate a batch)
    img = np.stack([mpimg.imread('res/elephant.jpg')])
    
    detect(img)
