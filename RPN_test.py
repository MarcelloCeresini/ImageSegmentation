# TODOs:
# TODO: Add training code
# TODO: Add the rest of Mask-RCNN
# TODO: Check if logits are REALLY needed (they may be useless debug-only tensors)
# TODO: Fix some comments in functions (mostly add output shapes and stuff)

import math
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle

import tensorflow as tf
# Refer to TF's internal version of Keras for more stability
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
import tensorflow.keras.backend as K

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
# Batch size for training and testing
BATCH_SIZE = 1

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
# Scales for the anchor boxes. They represent the square-rooted area
# of the anchor (so, a scale of 32 means that the anchor has an area of
# 32^2 pixels). Mathematically, they are sqrt(width*height).
RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
# Ratios of anchors at each cell (width/height)
# A value of 1 represents a square anchor, and 2 is a wide anchor
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

# The RPN generates a high amount of ROIs by sliding anchors over the 
# feature maps and calculating adjustments. We can reduce the amount
# of ROIs via the NMS algorithm.
# ROIs kept before non maximum suppression when ordering them by score
# (for performance reasons)
PRE_NMS_LIMIT = 6000
# ROIs kept after non-maximum suppression (training and inference)
POST_NMS_ROIS_TRAINING = 2000
POST_NMS_ROIS_INFERENCE = 1000
# Non-max suppression threshold to filter RPN proposals.
# Can be increased during training to generate more propsals.
RPN_NMS_THRESHOLD = 0.7


# Anchor cache: when dealing with images of the same shape, we don't want
# to calculate anchor coordinates over and over again, thus we mantain
# a cache
ANCHOR_CACHE = {}

# Execution mode: Training or Evaluation
EXECUTION_MODE = 'evaluation' # or 'training

##########################
### NETWORK DEFINITION ###
##########################

################
### BACKBONE ###
################

class InvalidBackboneError(Exception):
    pass

def resnet_graph(input_image):
    # Return the 5 shared convolutional layers
    if BACKBONE_NETWORK == 'resnet101':
        model = tf.keras.applications.ResNet101(
            include_top=False, # set to False to remove the classifier
            weights='imagenet', 
            input_tensor=input_image, 
            pooling=None, # DON'T apply max pooling to last layer
        )
        C1 = model.get_layer('pool1_pool').output
        C2 = model.get_layer('conv2_block3_out').output
        C3 = model.get_layer('conv3_block4_out').output
        C4 = model.get_layer('conv4_block23_out').output
        C5 = model.get_layer('conv5_block3_out').output
    elif BACKBONE_NETWORK == 'resnet50':
        model = tf.keras.applications.ResNet50(
            include_top=False, # set to False to remove the classifier
            weights='imagenet', 
            input_tensor=input_image, 
            pooling=None, # DON'T apply max pooling to last layer
        )
        C1 = model.get_layer('pool1_pool').output
        C2 = model.get_layer('conv2_block3_out').output
        C3 = model.get_layer('conv3_block4_out').output
        C4 = model.get_layer('conv4_block6_out').output
        C5 = model.get_layer('conv5_block3_out').output
    else:
        raise(InvalidBackboneError('The selected backbone is not yet supported'))
    return [C1,C2,C3,C4,C5]

###########
### RPN ###
###########

def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    '''Builds the actual computation graph of the RPN.

    Inputs: 
        - feature_map: backbone features [batch, height, width, depth]
        - anchors_per_location: number of anchors per pixel in the feature map
        - anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).

    Outputs:
        - rpn_class_logits: [batch, H * W * anchors_per_location, 2] 
            Anchor classifier logits (before softmax)
        - rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
        - rpn_deltas: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
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
    # It's just a 1x1 convolution because we only need to create scores without applying
    # spatial transformations.
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

    # Compute a bounding box refinement
    # The output of this layer will be a [batch, H, W, anchors per location * 4] tensor
    # meaning that for each pixel of the previous feature map (H,W) we will have the anchors
    # we wanted (anchors_per_location), each described by 4 numbers.
    # These 4 numbers are actually:
    # dx,dy: the refinement to apply to the center of the anchor
    # log(dw), log(dh): the (log-space) refinements of width and height of the anchor
    # The refinement will transform anchor boxes' center coordinates (ax, ay) and 
    # width (aw) and height (ah) like so:
    # 
    # fx = ax + dx*aw (these are deltas that need to be scaled with the actual anchor measures)
    # fy = ay + dy*ah
    # fh = ah * e^(log(dh)) = ah * dh (instead here we use the delta to scale measures directly)
    # fw = aw * dw
    x = KL.Conv2D(anchors_per_location*4, (1,1), padding='valid',
                    activation='linear', name='rpn_deltas_pred')(shared)
    
    ## As done before, we reshape this output to [batch, anchors, 4]
    rpn_deltas = KL.Lambda(
        lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4])
    )(x)

    # Return the obtained tensors
    return [rpn_class_logits, rpn_probs, rpn_deltas]


def build_rpn_model(anchor_stride, anchors_per_location, depth):
    '''Builds a Keras model for the RPN.

    Inputs: 
    - anchors_per_location: the number of anchors per pixel in the feature map.
        Usually this number corresponds with the number of possible ratios of
        anchors
    - anchor_stride: the stride to apply to the anchors generation. 1 means: generate
        one anchor per pixel in the feature map, while 2 means one anchor
        every 2 pixels.
    - depth: depth of the backbone feature map

    (Remember that each proposal is classified in one of two classes, namely
    foreground and background)

    Outputs:
        - a Keras Model, which itself outputs:
            - rpn_class_logits: [batch, H * W * anchors_per_location, 2] 
                Anchor classifier logits (before softmax)
            - rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
            - rpn_deltas: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                        applied to anchors.
    '''
    input_feature_map = KL.Input(shape=[None,None, depth],
                            name='input_rpn_feature_map')
    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
    return KM.Model([input_feature_map], outputs, name='rpn_model')

###########################
# NMS and Bbox refinement #
###########################

def apply_box_deltas_batched(boxes, deltas):
    '''
    Applies the given tensor of deltas to the given tensor of boxes

    Inputs:
    - boxes: [B, N, (y1,x1,y2,x2)] anchor boxes to update
    - deltas: [B, N, (dy,dx,log(dh),log(dw))] refinements to apply to the boxes

    Returns:
    - result: [B, N, (ny1,nx1,ny2,nx2)] the refined boxes
    '''
    # Get all important values from the tensors
    heights = boxes[:, :, 2] - boxes[:, :, 0]
    widths  = boxes[:, :, 3] - boxes[:, :, 1]
    center_y = boxes[:, :, 0] + heights*0.5
    center_x = boxes[:, :, 1] + widths *0.5
    # Apply deltas 
    center_y = center_y + deltas[:, :, 0] * heights
    center_x = center_x + deltas[:, :, 1] * widths
    heights  = heights * tf.exp(deltas[:, :, 2])
    widths   = widths  * tf.exp(deltas[:, :, 3])
    # Convert back into y1,x1,y2,x2 coordinates
    y1 = center_y - heights*0.5
    y2 = y1 + heights
    x1 = center_x - widths*0.5
    x2 = x1 + widths
    # Stack the measures in a new tensor
    return tf.stack([y1,x1,y2,x2], axis=2, # Axis = 2 ensures that we stack on the inner
                                           # dimension, so that we obtain a batch with
                                           # the desired number of 4-elements tensors.
                    name='apply_box_deltas')

def clip_boxes_batched(boxes, window):
    """
    Clips the tensor of boxes into the extremes defined in window

    Inputs:
    - boxes [B, N, (y1,x1,y2,x2)]
    - window [4] in the form y1, x1, y2, x2: they represent the boundaries of the image
    """
    # Split the tensors for ease of use
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1 = boxes[:,:,0]
    x1 = boxes[:,:,1]
    y2 = boxes[:,:,2]
    x2 = boxes[:,:,2]
    # Clip
    # To ensure that any coordinate is in the defined range, we must:
    # - Select the minimum between the coordinate and the maximum boundary
    # - Select the maximum between this and the minimum boundary
    # For example, when dealing with xs the first condition ensures that the coordinate
    # doesn't go out of boundary on the right and the second that it doesn't go out of
    # boundary on the left.
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.stack([y1,x1,y2,x2], axis=2, name="clipped_boxes")
    return clipped

class RefinementLayer(KL.Layer):
    '''
    Receives anchor scores and selects a subset to pass as proposals
    to the second part of the architecture.
    - Applies bounding box refinement deltas to anchors
    - Applies Non-Maximum suppression to limit overlaps between anchors

    Inputs:
    - A tensor containing 3 different tensors:
        - rpn_probs : [batch, num_anchors, (bg prob, fg prob)]
        - rpn_deltas: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
        - anchors: [batch, num_anchors, (y1, x1, y2, x2)] anchors in 
        normalized coordinates

    Outputs:
    - Refined proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    '''
    def __init__(self, proposal_count, nms_threshold, **kwargs):
        super(RefinementLayer, self).__init__(**kwargs)
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold

    def call(self, inputs):
        '''
        Entry point for the layer call.
        '''
        # Box foreground scores
        # Keep batch and num_anchors, ignore background probability
        scores = inputs[0][:, :, 1] 
        # Box deltas to be applied to the anchors
        deltas = inputs[1]
        # https://github.com/matterport/Mask_RCNN/issues/270#issuecomment-367602954
        # In the original Matterport and also Faster-RCNN implementation, 
        # the training targets are normalized because empirically it was found
        # that regressors work better when their output is normalized, so
        # when their mean is 0 and standard deviation is 1.
        # To achieve this, the target are transformed by:
        # 1) subtracting the mean of their coordinates
        # 2) dividing by the standard deviation
        # This means that the regressor outputs normalized coordinates,
        # so at this point deltas are normalized. To get back to real coordinates,
        # we should add the mean of the coordinates and multiply by the stdevs.
        # Since deltas are distributed between positive and negative, we can
        # assume that the mean is 0 and skip the first operation. The second operation
        # is the one depicted below. Standard deviations of the coordinates are 
        # precomputed or made up: the important thing is that they are kept consistent
        # within testing and training.
        # Uncomment and define a global constant keeping the STD_DEVs if needed.
        # deltas *= np.reshape(np.array([0.1, 0.1, 0.2, 0.2]), [1, 1, 4])

        # Anchors
        anchors = inputs[2]

        # Instead of applying refinements and NMS to all anchors (>20000 sometimes)
        # for performance we can trim the set of refined anchors by taking the top
        # k elements (ordering by scores of foreground-ness).
        # If there are less than the number we have chosen, take all anchors instead
        pre_nms_limit = tf.minimum(PRE_NMS_LIMIT, tf.shape(anchors)[1])
        # This function returns both values and indices, but we only need the indices
        top_indexes = tf.math.top_k(scores, k=pre_nms_limit, sorted=True,
                                name='top_anchors_by_score').indices

        # Reduce also scores and deltas tensors
        # TODO: I changed this part a lot, check that tensors shapes are the same of the other code
        # Gather lets us index the scores array with a tensor of indices (top indexes).
        # Since we can have multiple images in our batch (scores and top_indexes are both
        # bi-dimensional array for this reason), batch_dims=1 tells the gather function to
        # apply the first batch of indices to the first batch of scores/deltas/anchors, the
        # second to the second, etc.
        # https://www.tensorflow.org/api_docs/python/tf/gather#batching
        scores = tf.gather(scores, top_indexes, batch_dims=1)
        deltas = tf.gather(deltas, top_indexes, batch_dims=1)
        pre_nms_anchors = tf.gather(anchors, top_indexes, batch_dims=1)

        # Apply deltas to the anchors to get refined anchors.
        # Note: at this point, boxes is a [G,N,4] tensor, G being the elements in the batch,
        # N being 6000 (or the number of pre-nms anchors). 
        # We need to do apply the deltas for every item in the batch.
        boxes = apply_box_deltas_batched(pre_nms_anchors, deltas)
        # Clip to image boundaries (in normalized coordinates, clip in 0..1 range)
        window = np.array([0,0,1,1], dtype=np.float32)
        boxes = clip_boxes_batched(boxes, window)
        return boxes, top_indexes

    def compute_output_shape(self, input_shape):
        return (None, PRE_NMS_LIMIT, 4) # TODO Change this with self.proposal_count after nms implementation

##########################
# COMPOSITION OF MODULES #
##########################

class BadImageSizeException(Exception):
    pass

def build(mode):
    """
    Builds the Backbone + RPN model.
    """
    h, w = IMAGE_SHAPE[:2]
    if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
        raise BadImageSizeException("Image size must be dividable by 2 at least 6 times "
                                    "to avoid fractions when downscaling and upscaling."
                                    "For example, use 256, 320, 384, 448, 512, ... etc. ")
            
    # Define inputs
    # a. The input image
    input_image = KL.Input(
        shape = IMAGE_SHAPE, name='input_image'
    )
    # b. The anchors in NORMALIZED coordinates
    input_anchors = KL.Input(
        shape = [None, 4], name='input_anchors'
    )

    # If we are training the network, we need the groundtruth rpn matches (1 or 0) 
    # and bounding boxes as well as the detections groundtruth 
    # (class IDs, bounding boxes and masks) as additional inputs
    if mode == 'training':
        # RPN
        input_rpn_match = KL.Input(
            shape = [None, 1], name='input_rpn_match', dtype = tf.int32
            # TODO: can we use int8 or a boolean for optimization?
        )
        input_rpn_bbox = KL.Input(
            shape = [None, 4], name = 'input_rpn_bbox', dtype = tf.float32
        )

        # Detections
        input_gt_class_ids = KL.Input(
            shape = [None], name = 'input_gt_class_ids', dtype = tf.int32
        )
        # Zero-padded GT boxes in pixels
        # [batch, MAX_GT_INSTANCES, (y1,x1,y2,x2)] (in pixels)
        input_gt_boxes = KL.Input(
            shape = [None], name = 'input_gt_boxes', dtype = tf.int32
        )
        # Normalize input coordinates
        gt_boxes = KL.Lambda(lambda x: norm_boxes_tf(
            x, K.shape(input_image)[1:3]) # Ignore batch and other measures
        )(input_gt_boxes)

        # Groundtruth masks in zero-padded pixels
        # [batch, height, width, MAX_GT_INSTANCES]
        # There is a limit to how many GROUNDTRUTH instances can be passed
        input_gt_masks = KL.Input(
            # Masks are as big as the image in height and width and there is one channel
            # for each mask. Each mask is boolean.
            shape=[IMAGE_SHAPE[0], IMAGE_SHAPE[1], None],
            name="input_gt_masks", dtype=bool
        )

    # Backbone: Bottom-up ResNet101 + Top-down FPN with
    # shared convolutional layers. 
    # We use the Keras pre-trained implementation of the ResNet101 backbone
    # from which we can extract the feature maps we need in the following
    # modules.

    ### BOTTOM-UP RESNET50 ###
    # Recall that:
    # C1 = batchx256x256x64
    # C2 = batchx256x256x256
    # C3 = batchx128x128x512
    # C4 = batchx64x64x1024
    # C5 = batchx32x32x2048
    C1,C2,C3,C4,C5 = resnet_graph(input_image)

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
    # List of feature maps for the rest of the network
    mrcnn_feature_maps = [P2,P3,P4,P5]
    # The Rest of the network does not use P6

    # TODO: ADD TRAINING CODE REGARDING ANCHORS ETC.
    if mode == 'training':
        # Anchors are not passes as input in training mode
        anchors = get_anchors(IMAGE_SHAPE)
        # As in the testing preparation code, anchors must be replicated
        # in the batch dimension
        anchors = np.broadcast_to(anchors, (BATCH_SIZE,) + anchors.shape)
        # From TF documentation (https://www.tensorflow.org/guide/variable):
        # A TensorFlow variable is the recommended way to represent shared, 
        # persistent state your program manipulates.
        # A tf.Variable represents a tensor whose value can be changed 
        # by running ops on it. Specific ops allow you to read and modify 
        # the values of this tensor.
        # Basically, we need a layer that yields a tensor containing the anchors
        anchors = KL.Lambda(lambda x: tf.Variable(anchors), name='anchors')(input_image)
    elif mode == 'evaluation':
        # In testing mode, anchors are given as input to the network
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
    outputs = [[c[x] for c in layers_outputs] for x in range(len(layers_outputs[0]))]
    # Now, we want to concatenate the list of lists 
    output_names = ["rpn_class_logits", "rpn_classes", "rpn_deltas"]
    # Then, we concatenate all elements of the list as rows of three long tensors,
    # containing all logits, all class probabilities and all bounding boxes.
    outputs = [KL.Concatenate(axis=1, name=n)(list(o))
                for o,n in zip(outputs, output_names)]

    # Finally, extract all tensors 
    rpn_class_logits, rpn_classes, rpn_deltas = outputs

    # The output of the RPN must be transformed in actual proposals for the rest of
    # the network.
    proposal_count = POST_NMS_ROIS_INFERENCE if mode == 'evaluation'\
                else POST_NMS_ROIS_TRAINING
    # Call the RefinementLayer to do NMS of anchors and apply box deltas
    rpn_rois, indexes = RefinementLayer(
        proposal_count=proposal_count,
        nms_threshold=RPN_NMS_THRESHOLD,
        name='ROI_refinement')([rpn_classes, rpn_deltas, anchors])

    rpn_classes = tf.gather(rpn_classes, indexes, batch_dims=1)
    # Finally instantiate the Keras model and return it
    model = KM.Model(inputs=[input_image, input_anchors],
                     outputs=[rpn_classes, rpn_rois],
                     name='rpn')
    
    return model

###############################
### PREPROCESSING UTILITIES ###
###############################

def resize_image(image, min_dim, max_dim):
    '''
    Resizes an image by keeping the aspect ratio unchanged and using zero-padding
    to reshape it to a square.

    Inputs:
        - min_dim: the size of the smaller dimension
        - max_dim: ensures that the image's longest side doesn't exceed this value

    Outputs:
        - image: the resized image
        - window: (y1,x1,y2,x2): since padding might be inserted in the returned image,
            this window contains the coordinates of the image part in the full image.
            x2, y2 are not included, so the last "acceptable" line x2-1 and the last
            "acceptable" column is y2-1
        - scale: the scale factor used to resize the image
        - padding: type of padding added to the image [(top, bottom), (left, right), (0, 0)]
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

#####################
### PREPROCESSING ###
#####################

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
    
######################
### MISC UTILITIES ###
######################

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    Generate anchors for a given input shape

    Inputs: 
        - scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
        - ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
        - shape: [height, width] spatial shape of the feature map over which
                to generate anchors. It corresponds with the shape of one of the
                feature maps in the FPN (P2,P3,P4,P5,P6)
        - feature_stride: Stride of the feature map relative to the image in pixels.
        - anchor_stride: Stride of anchors on the feature map. For example, if the
            value is 2 then generate anchors for every other feature map pixel.
    
    Outputs:
        - boxes: #TODO: add the shape of this output list
    """
    # Get all possible combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Ratios are width/height, scales are sqrt(width*height).
    # To get actual widths and heights of the anchor boxes, we need to:
    # - Take the square root of the ratios, so sqrt(width)/sqrt(height)
    # - Multiply or divide these values with the scales in order to get
    #   either the width or the height
    #                 scales                  sqrt(ratios)
    #   width = sqrt(width)*sqrt(height)*sqrt(width)/sqrt(height) = 
    #   = sqrt(width)*sqrt(width) = sqrt(width)**2
    #
    #   Same for heights, but dividing for the square-rooted ratios so
    #   that nominator and denominator get inverted and we simplify the
    #   sqrt(width) term rather than the sqrt(height) which is our 
    #   "target"
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

    # The shapes of these arrays are:
    # Total number of feature maps pixels x 3 (the three scales for each anchor position)
    # Thus, we can reshape the arrays to get two lists: one of (y,x) and one of (h, w)
    box_centers = np.stack(
        # Stack them together in a new axis ([anchors, 3, 2])
        [box_centers_y, box_centers_x], axis=2
    ).reshape([-1, 2]) # Unstack anchors creating a [total_ancors, 2] array
    box_sizes = np.stack(
        [box_heights, box_widths], axis=2
    ).reshape([-1, 2])

    # Finally, convert the arrays into a single big array with (y1, x1, y2, x2) coordinates
    boxes = np.concatenate([box_centers - 0.5 * box_sizes, # y1, x1
                            box_centers + 0.5 * box_sizes], axis=1) # y2, x2
    # Concatenate on the columns obtaining x1,y1,x2,y2
    return boxes

def norm_boxes(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    
    Inputs: 
    - boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
    - shape: (height, width) in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Outputs:
        - [N, (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = shape
    # Shift x2 and y2 back into the image boundaries
    shift = np.array([0, 0, 1, 1])
    # ...thus reducing rows and columns by 1.
    scale = np.array([
        h - 1,
        w - 1,
        h - 1,
        w - 1
    ])
    return ((boxes - shift) / scale).astype(np.float32)

def norm_boxes_tf(boxes, shape):
    '''Same as the function above, but using tensorflow to deal with tensors
    '''
    shape = tf.cast(shape, tf.float32)  # Cast the shapes of the image to float32
    h, w = tf.split(shape, 2)           # Split in two sub-tensors
    scale = tf.concat([h,w,h,w], axis=-1) - tf.constant(1.0) # Concatenate h and w and reduce them all by 1
    shift = tf.constant([0.,0.,1.,1.])
    return tf.divide(boxes-shift, scale) 

# Just the inverse function
def denorm_boxes(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [N, (y1, x1, y2, x2)] in normalized coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in pixel coordinates
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.around((boxes * scale) + shift).astype(np.int32)

def denorm_boxes_tf(boxes, shape):
    '''
    Same as the function above, but using tensorflow operation to deal with tensors
    '''
    shape = tf.cast(shape, tf.float32)  # Cast the shapes of the image to float32
    h, w = tf.split(shape, 2)           # Split in two sub-tensors
    scale = tf.concat([h,w,h,w], axis=-1) - tf.constant(1.0)    # Concatenate h and w and reduce them all by 1
    shift = tf.constant([0.,0.,1.,1.])
    return tf.cast(tf.round(tf.multiply(boxes, scale) + shift), tf.int32) # Cast back into pixels

def get_anchors(image_shape):
    """
    Returns the anchor pyramid for the given image shape

    Inputs:
        - image_shape: the shape of the input image
    
    Outputs:
        - #TODO fill this with the output shape

    """
    global ANCHOR_CACHE
    backbone_shapes = np.array([
        [int(math.ceil(image_shape[0] / stride)),
         int(math.ceil(image_shape[1] / stride))]
         for stride in BACKBONE_STRIDES]
    )
    if not tuple(image_shape) in ANCHOR_CACHE:
        # If we have not calculated the anchor coordinates for an image with the same shape,
        # do and save them
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
        # Normalize anchors coordinates
        anchors = norm_boxes(anchors, image_shape[:2])
        ANCHOR_CACHE[tuple(image_shape)] = anchors
    return ANCHOR_CACHE[tuple(image_shape)]

##########################
### DETECTION PIPELINE ###
##########################

def detect(images, model: KM.Model):
    '''
    This function runs the detection pipeline.

    Inputs: 
    - images: a list of images, even of different sizes 
        (they will be reshaped as zero-padded squares of the same dimensions)
    - model: the model to run on the image (passed as input because it's easier
        for testing)

    Outputs:
        - preprocessed_images: the preprocessed images in a batch
        - anchors: the anchors for the image
        - rpn_classes: the classes (fg/bg) predicted by the RPN and their probabilities
        - rpn_boxes: the boxes predicted by the RPN
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
    # The original matterport implementation comments the following action
    # saying that Keras requires it. Basically, anchors are replicated among
    # the batch dimension. In our case, batch size is simply one.
    anchors = np.broadcast_to(anchors, (BATCH_SIZE,) + anchors.shape)

    # Use the previously instantiated model to run prediction
    rpn_classes, rpn_bboxes = model.predict([preprocessed_images, anchors])

    # Return the preprocessed images, anchors, classifications and bounding boxes from the RPN
    return preprocessed_images, rpn_classes, rpn_bboxes

###########################
### MAIN (TESTING CODE) ###
###########################

if __name__ == "__main__":
    '''
    Entry point for the code
    '''
    model = build(EXECUTION_MODE)
    # We need to compile the model before using it

    # Test the detection with one image (stack it n times to simulate a batch)
    img = np.stack([mpimg.imread('res/elephant.jpg')]*BATCH_SIZE)
    mod_images, rpn_classes, rpn_bboxes = detect(img, model)

    print("Shape of rpn_classes: {}".format(tf.shape(rpn_classes)))
    print("Shape of rpn_bboxes: {}".format(tf.shape(rpn_bboxes)))

    BBOXES_TO_DRAW = 50

    # Show each image sequentially and draw a selection of "the best" RPN bounding boxes.
    # Note that the model is not trained yet so "the best" boxes are really just random.
    for i in range(len(mod_images)):
        image = mod_images[i, :, :]
        classes = rpn_classes[i, :, :]
        bboxes = rpn_bboxes[i, :, :]
        # Select positive bboxes
        condition = np.where(classes[:, 0] > 0.5)[0]
        # If there is at least a positive bbox, draw it, otherwise draw random ones
        if len(condition): 
            bboxes = bboxes[condition]
        # Sort by probability
        rnd_bboxes = sorted(np.arange(0, bboxes.shape[0], 1),
                            key=lambda x, c=classes: c[x, 1])[:BBOXES_TO_DRAW]
        rnd_bboxes = bboxes[rnd_bboxes, :]
        rnd_bboxes = denorm_boxes(rnd_bboxes, image.shape[:2])
        fig, ax = plt.subplots()
        # Note that the image was previously normalized so colors will be weird
        ax.imshow(image)
        for bb in rnd_bboxes:
            rect = Rectangle(
                (bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1],
                linewidth=1, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)
        plt.show()