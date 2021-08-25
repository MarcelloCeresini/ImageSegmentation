# TODO: ADD TRAINING CODE
# TODO: Add the rest of Mask-RCNN
# TODO: Fix some comments in functions (mostly add output shapes and stuff)
# TODO: Possibly change the way some layers/functions work

import math
import numpy as np
import re
import datetime
import os
import errno
import multiprocessing

import tensorflow as tf
# Refer to TF's internal version of Keras for more stability
from tensorflow import keras
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
import tensorflow.keras.backend as K
import tensorflow.keras.losses as KLS

import config
import utils_functions as utils
from config import ModelConfig
from data_generator import DataGenerator

### GENERAL ERROR CLASSES ###

class BadImageSizeException(Exception):
    pass

class InvalidBackboneError(Exception):
    pass

#######################
### RESNET BACKBONE ###
#######################

def resnet_graph(input_image, backbone_type):
    '''
    Sends the input image through the ResNet backbone (101 or 50, depending
    on the chosen backbone type).
    The backbone is always initialized with imagenet-trained weights.
    We return the 5 convolutional layers that are shared between the backbone
    and the rest of the network (namely, the FPN).
    '''
    if backbone_type == 'resnet101':
        model = tf.keras.applications.ResNet101(
            include_top=False,  # set to False to remove the classifier
            weights='imagenet', # auto-download imagenet-trained weights
            input_tensor=input_image,
            pooling=None,  # DON'T apply max pooling to last layer
        )
        C1 = model.get_layer('pool1_pool').output
        C2 = model.get_layer('conv2_block3_out').output
        C3 = model.get_layer('conv3_block4_out').output
        C4 = model.get_layer('conv4_block23_out').output
        C5 = model.get_layer('conv5_block3_out').output
    elif backbone_type == 'resnet50':
        model = tf.keras.applications.ResNet50(
            include_top=False,  # set to False to remove the classifier
            weights='imagenet', # auto-download imagenet-trained weights
            input_tensor=input_image,
            pooling=None,  # DON'T apply max pooling to last layer
        )
        C1 = model.get_layer('pool1_pool').output
        C2 = model.get_layer('conv2_block3_out').output
        C3 = model.get_layer('conv3_block4_out').output
        C4 = model.get_layer('conv4_block6_out').output
        C5 = model.get_layer('conv5_block3_out').output
    else:
        raise (InvalidBackboneError('The selected backbone is not yet supported'))
    return [C1, C2, C3, C4, C5]

##################
### RPN MODULE ###
##################

def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    """Builds the actual computation graph of the RPN.

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

    """
    # Make the feature map deeper
    # The result is the convolutional layer on which the RPN will evaluate anchors
    shared = KL.Conv2D(512, (3, 3), padding='same', activation='relu',
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
    x = KL.Conv2D(2 * anchors_per_location, (1, 1), padding='valid', activation='linear',
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
    x = KL.Conv2D(anchors_per_location * 4, (1, 1), padding='valid',
                  activation='linear', name='rpn_deltas_pred')(shared)

    ## As done before, we reshape this output to [batch, anchors, 4]
    rpn_deltas = KL.Lambda(
        lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4])
    )(x)

    # Return the obtained tensors
    return [rpn_class_logits, rpn_probs, rpn_deltas]

def build_rpn_model(anchor_stride, anchors_per_location, depth):
    """Builds a Keras model for the RPN.

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
    """
    input_feature_map = KL.Input(shape=[None, None, depth],
                                 name='input_rpn_feature_map')
    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
    return KM.Model([input_feature_map], outputs, name='rpn_model')

###############################
# NMS & BBOX REFINEMENT LAYER #
###############################

def apply_box_deltas_batched(boxes, deltas):
    """
    Applies the given tensor of deltas to the given tensor of boxes

    Inputs:
    - boxes: [B, N, (y1,x1,y2,x2)] anchor boxes to update
    - deltas: [B, N, (dy,dx,log(dh),log(dw))] refinements to apply to the boxes

    Returns:
    - result: [B, N, (ny1,nx1,ny2,nx2)] the refined boxes
    """
    # Get all important values from the tensors
    heights = boxes[:, :, 2] - boxes[:, :, 0]
    widths = boxes[:, :, 3] - boxes[:, :, 1]
    center_y = boxes[:, :, 0] + heights * 0.5
    center_x = boxes[:, :, 1] + widths * 0.5
    # Apply deltas 
    center_y = center_y + deltas[:, :, 0] * heights
    center_x = center_x + deltas[:, :, 1] * widths
    heights = heights * tf.exp(deltas[:, :, 2])
    widths = widths * tf.exp(deltas[:, :, 3])
    # Convert back into y1,x1,y2,x2 coordinates
    y1 = center_y - heights * 0.5
    y2 = y1 + heights
    x1 = center_x - widths * 0.5
    x2 = x1 + widths
    # Stack the measures in a new tensor
    return tf.stack([y1, x1, y2, x2], axis=2,  # Axis = 2 ensures that we stack on the inner
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
    y1 = boxes[:, :, 0]
    x1 = boxes[:, :, 1]
    y2 = boxes[:, :, 2]
    x2 = boxes[:, :, 3]
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
    clipped = tf.stack([y1, x1, y2, x2], axis=2, name="clipped_boxes")
    return clipped

class RefinementLayer(KL.Layer):
    """
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
    """

    def __init__(self, proposal_count, config:ModelConfig, **kwargs):
        super(RefinementLayer, self).__init__(**kwargs)
        self.proposal_count = proposal_count
        self.config = config

    def get_config(self):
        '''
        To be able to save the model we need to update the configuration
        for this custom layer by adding the parameters in init.
        '''
        config = super().get_config().copy()
        config.update({
            'proposal_count': self.proposal_count,
            'config': self.config
        })
        return config

    def call(self, inputs, **kwargs):
        """
        Entry point for the layer call.
        """
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
        pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(anchors)[1])
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
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        ### Probably cllipping isn't needed, uncomment if needed
        # boxes = clip_boxes_batched(boxes, window)
        # Apply non maximum suppression using tensorflow's implementation of a batched NMS
        nmsed_boxes, nmsed_scores, _, _ = tf.image.combined_non_max_suppression(
                                                tf.expand_dims(boxes, 2),
                                                tf.expand_dims(scores,2),
                                                self.proposal_count, self.proposal_count,
                                                self.config.RPN_NMS_THRESHOLD)
        # The original code adds padding to these tensors, in case the self.proposal_count
        # requirement is not respected, but this is only required when dealing with very 
        # small images. I think we are fine without it, for now.
        return nmsed_boxes, nmsed_scores

    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)

######################
# RPN TRAINING LAYER #
######################

class DetectionTargetLayer(KL.Layer):
    """
    Takes a subset of RPN's proposals so that negative proposals don't overshadow 
    positive ones. Then, associates each proposals to either the background or a
    target GT bounding box. Finally, it prepares each of these selected proposals by
    generating box refinements (distance from the GT box), class_ids and object masks
    (by taking them from the appropriate GT object). 
    Proposals enriched with this information will be used as training samples.

    For initialization, the ModelConfig object is required.
    
    For calls:

    Inputs:
    - proposals: [batch, N, (y1,x1,y2,x2)] in normalized coordinates.
        If there are not enough proposals, it might be a zero-padded array.
    - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer representing class IDs.
    - gt_boxes: [batch, MAX_GT_INSTANCES, (y1,x1,y2,x2)] in normalized coordinates.
    - gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type, representing
        where the groundtruth boxes are on the image pixels.

    Returns:
    - rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1,x1,y2,x2) in normalized coordinates]
    - target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE] Integers representing class IDs.
    - target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)]
    - target_masks: [batch, TRAIN_ROIS_PER_IMAGE, height, width]
                 Masks cropped to bbox boundaries and resized to neural
                 network output size.

    If there are not enough proposals (eg. the proposals array is heavily zero-padded,
    the target ROIs can also be zero padded)
    """
    def __init__(self, config:ModelConfig, name):
        super(DetectionTargetLayer, self).__init__(name=name)
        self.config = config

    def get_config(self):
        '''
        To be able to save the model we need to update the configuration
        for this custom layer by adding the parameters in init.
        '''
        config = super().get_config().copy()
        config.update({
            'config': self.config
        })
        return config

    def call(self, inputs):
        # We need to slice the batch and run a graph for each slice, because
        # the number of non-zero padded elements in tensors can be different
        # within the batch.
        out_names = ["rois", "target_class_ids", "target_bbox", "target_mask"]
        outputs = []
        for i in range(self.config.BATCH_SIZE):
            inputs_slice = [x[i] for x in inputs]
            output_slice = self.detection_targets_graph(*inputs_slice)
            outputs.append(output_slice)
        # Them, change from a list of slices containing outputs to
        # a list of outputs containing slices.
        # In other words, group together outputs by their significance rather than
        # their position in the batch
        outputs = list(zip(*outputs))
        # Stack in a single tensor this list of lists, giving the desired name
        # to each tensor.
        result = [tf.stack(o, axis=0, name=n)
                    for o, n in zip(outputs, out_names)]
        return result

    @tf.function
    def detection_targets_graph(self, proposals, gt_class_ids, gt_boxes, gt_masks):
        """
        See documentation for the layer for the explanation of the inputs to this graph.
        Consider that each tensor here is not batched, so, for instance, the shape of
        proposals is [N, 4].
        """
        # Make some assertion checks for the rest of the layer to work properly
        asserts = [
            tf.Assert(tf.greater(tf.shape(proposals[1]), 0),
            [proposals], name='roi_assertion') # Prints out the proposals tensor if condition is false
        ]
        with tf.control_dependencies(asserts):
            proposals = tf.identity(proposals) # Copy the tensor in another tensor

        ## CLEANING FROM PADDING ##
        
        # Remove zero paddings from proposals and gt_boxes
        proposals, _ = trim_zeros_tf(proposals, name='trim_proposals')
        gt_boxes, non_zeros = trim_zeros_tf(gt_boxes, name='trim_gt_boxes')
        # Use the mask of gt_boxes to select GT class IDs and masks
        gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                        name='trim_gt_class_ids')
        gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:,0], 
                                axis=2, name="trim_gt_masks")

        ## CLEANING FROM CROWDS ##

        # Handle COCO crowds
        # A crowd box in COCO is a bounding box around several instances.
        # We exclude them from training. To recognize them, we check
        # for negative class IDs, since that's what is assigned to crowds
        # (see the FoodDataset in food.py)
        # TODO: is this actually good given our dataset?
        crowd_ix = tf.where(gt_class_ids < 0)[:,0]
        non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
        crowd_boxes = tf.gather(gt_boxes, crowd_ix)
        gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
        gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
        gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)

        ## COMPUTING OVERLAPS AND SELECTING BEST MATCHES BASED ON THEM ##

        # Now we need to compute overlaps, that is, understand where and in what
        # measure do our proposals match groundtruth boxes. To do this we use the
        # IoU measure (intersection over union or Jaccard index).
        overlaps = calculate_overlaps_matrix_tf(proposals, gt_boxes)
        # overlaps will be a NxN matrix of IoUs.
        # We also calculate the overlaps with the crowds boxes, obtaining a matrix
        # that is [proposals, crowd_boxes].
        crowd_overlaps = calculate_overlaps_matrix_tf(proposals, crowd_boxes)
        # Get the highest IoU for each row (the best groundtruth box for each proposal)
        crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
        # Where the intersection over union with the best groundtruth box is very small,
        # we can say that there is not an interesting crowd box for that proposals. Thus,
        # create an array selecting which "best proposals" are crowds and which aren't.
        no_crowd_mask = (crowd_iou_max < 0.001)

        ## ASSIGNING FG/BG INDEX ##

        # Now, remember that some RoIs from the RPN are positive while some are negative.
        # We must determine which are which in the same way as above:
        # 1: Get the best IoU from each row
        roi_iou_max = tf.reduce_max(overlaps, axis=1)
        # So we create two boolean masks:
        # 1) positives:
        positive_roi_mask = (roi_iou_max >= 0.5)
        positive_indices = tf.where(positive_roi_mask)[:,0]
        # 2) negatives: we consider both max IoUs inferior to 0.5 and crowd RoIs.
        negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_mask))[:,0]

        ## SUBSAMPLE PROPOSALS BY KEEPING A GOOD RATIO ##

        # Now, we need to face a problem that is also mentioned in the original Faster-RCNN paper.

        # "It is possible to optimize
        # for the loss functions of all anchors, but this will
        # bias towards negative samples as they are dominate.
        # Instead, we randomly sample 256 anchors in an image
        # to compute the loss function of a mini-batch, where
        # the sampled positive and negative anchors have a
        # ratio of up to 1:1. If there are fewer than 128 positive
        # samples in an image, we pad the mini-batch with
        # negative ones." 
        
        # Keep in mind that in that implementation, the model was trained using 512 proposals per image, 
        # while we only keep 256 to avoid excessive padding in cases where not enough proposals are
        # generated. Therefore, we choose to go for a safer "subsampling 33% percent of the total positive 
        # subsamples" and balance with negative proposals to keep a 1:2 ratio between positives and negatives.
        positive_count = int(self.config.TRAIN_ROIS_PER_IMAGE * self.config.ROI_POSITIVE_RATIO)
        positive_indices = tf.random.shuffle(positive_indices)[:positive_count]
        positive_count = tf.shape(positive_indices)[0]
        # How many negatives do we need exactly? 
        # The ratio is 0.3 = positive/negative, so negative = 1/0.33 * positive
        negative_count = tf.cast(1.0 / self.config.ROI_POSITIVE_RATIO * tf.cast(positive_count, tf.float32), tf.int32)
        # To keep it 1:2:
        negative_count -= positive_count
        negative_indices = tf.random.shuffle(negative_indices)[:negative_count]
        # Gather selected ROIs
        positive_rois = tf.gather(proposals, positive_indices)
        negative_rois = tf.gather(proposals, negative_indices)

        ## MATCH SELECTED POSITIVE ROIS WITH THEIR GT BBOX ##

        # Assign positive RoIs to GT boxes gathering results from the overlaps matrix
        positive_overlaps = tf.gather(overlaps, positive_indices)
        # Since there might be no GT boxes for the image and the overlaps matrix can be empty,
        # we check that positive_overlaps contains row that are longer than the empty tensor.
        roi_gt_box_assignment = tf.cond(
            tf.greater(tf.shape(positive_overlaps)[1], 0),
            # If the check succeeds, we the groundtruth box becomes the position of the element in the row
            # that has the larger value
            true_fn = lambda: tf.argmax(positive_overlaps, axis=1),
            # Otherwise, we return an empty tensor.
            false_fn = lambda: tf.cast(tf.constant([]),tf.int64)
        )
        # We then gather gt_boxes and classes using the map above.
        roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
        roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

        ## CALCULATE BB REFINEMENTS BETWEEN GT BBOX AND ASSOCIATED PROPOSALS ##

        # What is the difference between the GT boxes and the proposals in terms of deltas? 
        # We need to compute the bounding box refinements for these ROIs
        box = tf.cast(positive_rois, tf.float32)
        gt_box = tf.cast(roi_gt_boxes, tf.float32)
        
        # Remember boxes are (y1,x1,y2,x2)
        height = box[:, 2] - box[:, 0]
        width = box[:, 3] - box[:, 1]
        center_y = box[:, 0] + 0.5 * height
        center_x = box[:, 1] + 0.5 * width

        # Same for GT
        gt_height = gt_box[:, 2] - gt_box[:, 0]
        gt_width = gt_box[:, 3] - gt_box[:, 1]
        gt_center_y = gt_box[:, 0] + 0.5 * gt_height
        gt_center_x = gt_box[:, 1] + 0.5 * gt_width

        # We calculate deltas as the difference between proposals and GTs
        # in the usual way, like box refinements are calculated
        dy = (gt_center_y - center_y) / height
        dx = (gt_center_x - center_x) / width
        dh = tf.math.log(gt_height / height)
        dw = tf.math.log(gt_width / width)

        deltas = tf.stack([dy, dx, dh, dw], axis=1)

        # Again, the deltas get divided by the STDEV in Matterport's implementation.
        # TODO: decide if it's a good idea
        # deltas /= config.BBOX_STD_DEV

        ## ASSOCIATE OBJECT MASKS TO PROPOSALS ##

        # Masks are currently in a [height, width, N] tensor.
        # Transpose masks to [N, height, width] and add a dimension at the end ([N, height, width, 1])
        transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
        # Pick the right mask for each ROI by using the same mask we have computed before.
        roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

        boxes = positive_rois
        # Give a unique ID to each mask
        box_ids = tf.range(0, tf.shape(roi_masks)[0])

        # Now we use the crop_and_resize function which is basically tensorflow's implementation of 
        # the ROIAlign method. 
        # TODO: Talk about ROIAlign and crop_and_resize.
        masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32),
                                         boxes, 
                                         box_ids,
                                         self.config.MASK_SHAPE)

        # Remove the extra dimension (depth) for the output
        masks = tf.squeeze(masks, axis=3)
        # Also, we want masks to be filled with 0 or 1 to use them in the binary crossentropy loss, 
        # so we need to threshold GT masks at 0.5
        # TODO: talk about this
        masks = tf.round(masks)

        # Finally, we can produce the output:
        # - rois are positive RoIs concatenated with negatives
        rois = tf.concat([positive_rois, negative_rois], axis=0)
        # We also need to pad this tensor and the following with zeroes in the positions that are not used
        # by negative RoIs
        N = tf.shape(negative_rois)[0] # Number of negative rois
        P = tf.maximum(self.config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0) # The number of positions to pad
        # About the padding function: it receives as input the tensor to pad and a [D,2] tensor, where D
        # is the rank of the first tensor (in our case, 2). For each dimension d, [d,0] and [d,1] indicate
        # how many values should we pad on before and after the contents of the tensor in dimension d.     
        rois = tf.pad(rois, [(0, P),(0, 0)]) # Add a padding of P elements on the right on the first dimension 
                                             # (the other dimension contains the coordinates and 
                                             # should not be changed)
        roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N+P), (0,0)]) # Negative ROIs are not included in GT
                                                               # boxes of course, so we need to pad also for them
        roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N+P)]) # This is a one dimensional tensor
        deltas = tf.pad(deltas, [(0,N+P),(0,0)])            # Deltas and masks of BG ROIs should not be included and be
        masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)]) # padded

        return rois, roi_gt_class_ids, deltas, masks

    def compute_output_shape(self, input_shape):
        return [
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4), # ROIs
            (None, self.config.TRAIN_ROIS_PER_IMAGE),    # class IDs
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4), # deltas
            (None, self.config.TRAIN_ROIS_PER_IMAGE, self.config.MASK_SHAPE[0], 
                                                     self.config.MASK_SHAPE[1]) # masks
        ]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None]

#################
### RPN CLASS ###
#################

class RPN():
    """
    This class encapsulates the RPN model and some of its functionalities.
    The Keras model is contained in the model property.
    """
    def __init__(self, mode:str, config:ModelConfig, out_dir:str=None):
        """
        Inputs: 
        - mode: One of ["training", "inference"]
        - config: The model configuration object (ModelConfig object)
        - out_dir (optional): Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.out_dir = out_dir
        if mode == 'training':
            self.set_log_dir()
        # Instantiate self.model:
        self.build()
        self.summary()

    def summary(self):
        print(self.model.summary())


    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            The path of the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = sorted([x for x in os.listdir(self.out_dir) if 
                        os.path.isdir(x) and 
                        x.startswith('food')])
        if not dir_names: # In case of empty list
            raise FileNotFoundError(
                errno.ENOENT,
                "Could not find model directory under {}".format(self.out_dir))
        # Pick last directory
        dir_name = os.path.join(self.out_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = sorted([x for x in os.listdir(dir_name) if 
                                not os.path.isdir(x) and    # Must be a weight file
                                x.startswith('rpn_food')])
        # If there are no valid checkpoints:
        if not checkpoints:
            raise FileNotFoundError(
                errno.ENOENT, "Could not find weight files in {}".format(dir_name))
        
        # Otherwise, return last checkpoint
        return os.path.join(dir_name, checkpoints[-1])


    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # \path\to\logs\food20211029T2315\rpn_food_0001.h5 (Windows)
            # /path/to/logs/food20211029T2315/rpn_food_0001.h5 (Linux)
            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]rpn\_food\_[\w-]+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                # Epoch number in file is 1-based, and in Keras code it's 0-based.
                # So, adjust for that then increment by one to start from the next epoch
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

        # Directory for training logs
        self.log_dir = os.path.join(self.out_dir, "{}{:%Y%m%dT%H%M}".format(
            'food', now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "rpn_food_*epoch*.h5")
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{self.epoch:04d}")


    def build(self):
        """
        Builds the Backbone + RPN model.
        """
        h, w = self.config.IMAGE_SHAPE[:2]
        if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
            raise BadImageSizeException("Image size must be dividable by 2 at least 6 times "
                                        "to avoid fractions when downscaling and upscaling."
                                        "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Define inputs
        # The first input is of course the input image
        input_image = KL.Input(
            shape=self.config.IMAGE_SHAPE, name='input_image'
        )

        # If we are training the network, we need the groundtruth rpn matches (1 or 0) 
        # and bounding boxes as well as the detections groundtruth 
        # (class IDs, bounding boxes and masks)
        if self.mode == 'training':
            # RPN
            # TODO: What are these two inputs exactly?
            input_rpn_match = KL.Input(
                shape=[None, 1], name='input_rpn_match', dtype=tf.int32
                # TODO: can we use int8 or a boolean for optimization?
            )
            input_rpn_bbox = KL.Input(
                shape=[None, 4], name='input_rpn_bbox', dtype=tf.float32
            )

            # Detections
            input_gt_class_ids = KL.Input(
                shape=[None], name='input_gt_class_ids', dtype=tf.int32
            )
            # Zero-padded GT boxes in pixels
            # [batch, MAX_GT_INSTANCES, (y1,x1,y2,x2)] (in pixels)
            input_gt_boxes = KL.Input(
                shape=[None, 4], name='input_gt_boxes', dtype=tf.float32
            )
            # Normalize input coordinates
            # Treating this with a Lambda layer makes the code crash, so we implemented a
            # custom layer. The original code is below:
            # gt_boxes = KL.Lambda(lambda x: norm_boxes_tf(
            #    x, tf.shape(input_image)[1:3])(input_gt_boxes)
            gt_boxes = NormBoxesLayer(name="norm_gt_boxes")([
                input_gt_boxes, input_image
            ])

            # Groundtruth masks in zero-padded pixels
            # [batch, height, width, MAX_GT_INSTANCES]
            # There is a limit to how many GROUNDTRUTH instances can be passed
            input_gt_masks = KL.Input(
                # Masks are as big as the image in height and width and there is one channel
                # for each mask. Each mask is boolean.
                shape=[self.config.IMAGE_SHAPE[0], self.config.IMAGE_SHAPE[1], None],
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
        C1, C2, C3, C4, C5 = resnet_graph(input_image, self.config.BACKBONE_NETWORK)

        ### TOP-DOWN FPN ###
        P5 = KL.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
        # P5 has shape 32x32x256
        P4 = KL.Add(name='fpn_p4add')([
            # UpSampling2D repeats rows and columns of the data (P5) 2 times.
            # Thus, this is 64x64x256
            KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
            # C4 is transformed into 64x64x256
            KL.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)
            # Hence the shapes match and we can perform an addition
        ])
        # P4 has shape 64x64x256
        P3 = KL.Add(name='fpn_p3add')([
            # 128x128x256
            KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
            # 128x128x256
            KL.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name="fpn_c3p3")(C3)
        ])
        # P3 has shape 128x128x256
        P2 = KL.Add(name='fpn_p2add')([
            # 256x256x256
            KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
            # 256x256x256
            KL.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name="fpn_c2p2")(C2)
        ])
        # P2 has shape 256x256x256
        # Attach 3x3 conv to all P layers to get the final feature maps.
        # All dimensions are kept the same
        P2 = KL.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="same", name="fpn_p2")(P2)
        P3 = KL.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="same", name="fpn_p3")(P3)
        P4 = KL.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="same", name="fpn_p4")(P4)
        P5 = KL.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="same", name="fpn_p5")(P5)
        # An additional feature map is generated by subsampling from P5
        # with stride 2
        P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name='fpn_p6')(P5)
        # P6 has shape 16x16x256

        # List of feature maps for the rpn
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        # List of feature maps for the rest of the network
        mrcnn_feature_maps = [P2, P3, P4, P5]
        # The Rest of the network does not use P6

        if self.mode == 'training':
            # Anchors are not passed as input in training mode
            # TODO: the mechanism that creates anchors needs to be fixed.
            #       We should use multiscale anchors!!
            anchors = self.get_anchors(self.config.IMAGE_SHAPE)
            # As in the testing preparation code, anchors must be replicated
            # in the batch dimension
            anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)
            # From TF documentation (https://www.tensorflow.org/guide/variable):
            # A TensorFlow variable is the recommended way to represent shared, 
            # persistent state your program manipulates.
            # A tf.Variable represents a tensor whose value can be changed 
            # by running ops on it. Specific ops allow you to read and modify 
            # the values of this tensor.
            # Basically, we need a layer that yields a tensor containing the anchors
            # TODO: Why not doing it with tensorflow directly?
            anchors = tf.Variable(anchors)
        elif self.mode == 'inference':
            # In testing mode, anchors are given as input to the network,
            # in NORMALIZED coordinates
            anchors = KL.Input(
                shape=[None, 4], name='input_anchors'
            )

        ### RPN MODEL ###
        # The RPN is a lightweight neural network that scans the image 
        # in a sliding-window fashion and finds areas that contain objects.
        # The regions that the RPN scans over are called anchors. 
        # Which are boxes distributed over the image area
        rpn_model = build_rpn_model(anchor_stride=self.config.RPN_ANCHOR_STRIDE,
                                    anchors_per_location=len(self.config.RPN_ANCHOR_RATIOS),
                                    depth=self.config.TOP_DOWN_PYRAMID_SIZE)

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
                for o, n in zip(outputs, output_names)]

        # Finally, extract all tensors 
        rpn_class_logits, rpn_classes, rpn_deltas = outputs

        # The output of the RPN must be transformed in actual proposals for the rest of
        # the network.
        proposal_count = self.config.POST_NMS_ROIS_INFERENCE if self.mode == 'inference' \
            else self.config.POST_NMS_ROIS_TRAINING

        # Call the RefinementLayer to do NMS of anchors and apply box deltas
        rpn_rois, rpn_classes = RefinementLayer(
            proposal_count=proposal_count,
            config=self.config,
            name='ROI_refinement')([rpn_classes, rpn_deltas, anchors])

        if self.mode == 'training':
            # We need to specify which are the class IDs the dataset supports
            # We haven't explored the dataset correctly yet, so we'll skip this
            # part for now

            # TODO: this is needed for the computation of the classification loss.
            #active_class_ids = KL.Lambda(
            #    lambda x: parse_image_meta_graph(x)["active_class_idx"]
            #)(input_image_meta)

            # Generate some target proposals among the set of ROIs we have generated
            # earlier in the network. These target proposals represent the target output
            # of the RPN for the image.
            rois, target_class_ids, target_bbox, target_mask = \
                DetectionTargetLayer(self.config, name="proposal_targets")([
                    rpn_rois, input_gt_class_ids, gt_boxes, input_gt_masks
                ])

            output_rois = tf.identity(rois, name="output_rois")

            # TODO: Here we should add the network heads: the classifier and the mask graph.

            # RPN losses:
            # 1. Compute loss for the classification BG/FG.
            rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])
            # 2. Compute loss for the bounding box regression.
            rpn_bbox_loss = KL.Lambda(lambda x: rpn_box_regression_loss_graph(*x), name="rpn_bbox_loss")(
                [self.config.BATCH_SIZE, input_rpn_bbox, input_rpn_match, rpn_deltas])

            # Model
            inputs = [input_image, input_rpn_match, input_rpn_bbox, input_gt_class_ids,
                input_gt_boxes, input_gt_masks]
            outputs = [rpn_class_logits, rpn_classes, rpn_deltas,
                rpn_rois, output_rois, rpn_class_loss, rpn_bbox_loss]

        elif self.mode == 'inference':
            inputs = [input_image, anchors]
            outputs = [rpn_classes, rpn_rois]

        # TODO: Add MaskRCNN's classification and regression heads.

        # Finally, instantiate the Keras model
        self.model = KM.Model(inputs, outputs, name='rpn')


    def detect(self, images):
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
        preprocessed_images, windows = utils.preprocess_inputs(images, self.config)

        # Check that all images in the batch now have the same size
        image_shape = preprocessed_images[0].shape
        for g in preprocessed_images[1:]:
            assert g.shape == image_shape, \
                "All images must have the same size after preprocessing"

        # The network also receives anchors as inputs, so we need a function
        # that returns the anchors
        anchors = self.get_anchors(image_shape)

        # The original matterport implementation comments the following action
        # saying that Keras requires it. Basically, anchors are replicated among
        # the batch dimension. In our case, batch size is simply one.
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        # Use the previously instantiated model to run prediction
        rpn_classes, rpn_bboxes = self.model.predict([preprocessed_images, anchors])

        # Return the preprocessed images, anchors, classifications and bounding boxes from the RPN
        return preprocessed_images, rpn_classes, rpn_bboxes


    def get_anchors(self, image_shape):
        """
        Returns the anchor pyramid for the given image shape

        Inputs:
            - image_shape: the shape of the input image
        
        Outputs:
            - #TODO fill this with the output shape
        """
        backbone_shapes = np.array([
            [int(math.ceil(image_shape[0] / stride)),
            int(math.ceil(image_shape[1] / stride))]
            for stride in self.config.BACKBONE_STRIDES]
        )
        # Anchor cache: when dealing with images of the same shape, we don't want
        # to calculate anchor coordinates over and over again, thus we mantain
        # a cache in this object. We initialize the cache at the first image.
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # If we have not calculated the anchor coordinates for an image with the same shape,
            # do and save them
            anchors = utils.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE)
            # Normalize anchors coordinates
            anchors = utils.norm_boxes(anchors, image_shape[:2])
            self._anchor_cache[tuple(image_shape)] = anchors
        return self._anchor_cache[tuple(image_shape)]


    def compile(self, learning_rate, momentum):
        '''
        Compile the model for training. This means setting the optimizer,
        the losses, regularization and others so that we have a train-ready model.

        # TODO: Documentation
        '''
        # Optimizer
        # We choose classic SGD as an optimizer.
        optimizer = keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=momentum,
            clipnorm=self.config.GRADIENT_CLIP_NORM
        )

        # TODO: does it really need to be so complicated?

        # Add losses
        loss_names = ["rpn_class_loss",  "rpn_bbox_loss"]
        for name in loss_names:
            # Retrieve the loss layer from the model
            layer = self.model.get_layer(name)
            # Apply mean within the batch
            loss = tf.math.reduce_mean(layer.output, keepdims=True)
            # Add loss
            self.model.add_loss(loss)
        
        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        # Note: from Keras's documentation:
        '''
        This method can also be called directly on a Functional Model during construction. 
        In this case, any loss Tensors passed to this Model must be symbolic and be able 
        to be traced back to the model's Inputs. 
        (...)
        If this is not the case for your loss (if, for example, 
        your loss references a Variable of one of the model's layers), 
        you can wrap your loss in a zero-argument lambda. 
        These losses are not tracked as part of the model's topology since they can't be serialized.
        '''
        self.model.add_loss(lambda: tf.add_n(reg_losses))

        # Compile the model
        self.model.compile(
            optimizer=optimizer,
            loss=[None] * len(self.model.outputs)
        )

        # Add metrics for losses
        for name in loss_names:
            if name in self.model.metrics_names:
                continue
            layer = self.model.get_layer(name)
            self.model.metrics_names.append(name)
            loss = tf.reduce_mean(layer.output, keepdims=True)
            self.model.metrics.append(loss)

    
    def set_trainable(self, layer_regex, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        layers = self.model.layers
        print("Selecting trainable layers in model")
        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                # Recursive call
                self.set_trainable(
                    layer, layer_regex, indent=indent + 4)
                continue

            # Does the layer have weights at all?
            if not layer.weights:
                continue

            # Is the layer trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainable layer names
            if trainable and verbose > 0:
                print("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))


    def train(self, train_dataset, val_dataset, 
                    learning_rate, epochs, layers, augmentation=None, 
                    custom_callbacks=None):
        """Train the model.
        train_dataset, val_dataset: Training and validation datasets.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done already, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting which layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
            heads: Only the RPN and FPN (#TODO: When the rest of the model is added, 
                                        this should train also the other heads)
            all: All the layers
            3+: Train Resnet stage 3 and up
            4+: Train Resnet stage 4 and up
            5+: Train Resnet stage 5 and up
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug)
            augmentation. For example, passing imgaug.augmenters.Fliplr(0.5)
            flips images right/left 50% of the time. You can pass complex
            augmentations as well. This augmentation applies 50% of the
            time, and when it does it flips images right/left half the time
            and adds a Gaussian blur with a random sigma in range 0 to 5.

                augmentation = imgaug.augmenters.Sometimes(0.5, [
                    imgaug.augmenters.Fliplr(0.5),
                    imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
                ])

        custom_callbacks: Optional. Add custom callbacks to be called
            with the keras fit_generator method. Must be list of type keras.callbacks.
        """
        assert self.mode == "training", "Create model in training mode."

        # Select the layers to train using some regex
        layer_regex = {
            # all layers but the backbone
            "heads": r"(rpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(conv3\_.*)|(conv4\_.*)|(conv5\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(conv4\_.*)|(conv5\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(conv5\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = DataGenerator(train_dataset, self.config, shuffle=True,
                                        augmentation=augmentation)
        val_generator = DataGenerator(val_dataset, self.config, shuffle=True,
                                        augmentation=augmentation)

        # Create log_dir if it does not exist
        os.makedirs(self.log_dir, exist_ok=True)

        # Callbacks
        # TODO: Should we add other callbacks?
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
        ]

        if custom_callbacks is not None:
            callbacks.extend(custom_callbacks)

        # Train
        print("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        print("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        # TODO: It's no harm, but check this out
        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name is 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        self.model.fit(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True,
        )
        self.epoch = max(self.epoch, epochs)

####################
### MODEL LOSSES ###
####################

@tf.function
def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    """
    Graph that calculates the RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for BG/FG.
    """
    # The classification loss is categorical crossentropy. This loss function
    # calculates the difference between the logits (which represent the probability
    # distribution of a RPN proposal to be FG or BG) and the actual class.
    # Note that the actual class can be positive (1), negative (-1) or neutral (0).
    # Neutral proposals should not contribute to the loss.

    # Squeeze the last dimension of the rpn_match to make things simpler
    # rpn_match becomes a [batch, anchors] tensor.
    rpn_match = tf.squeeze(rpn_match)
    # Select usable indices for the loss
    usable_indices = tf.where(K.not_equal(rpn_match, 0))
    # Filter the usable rows for the loss
    rpn_class_logits = tf.gather_nd(rpn_class_logits, usable_indices)
    anchor_class = tf.gather_nd(rpn_match, usable_indices)
    # Transform -1/1 in 0/1 for positive and negative in anchor_class
    anchor_class = K.cast(K.equal(anchor_class, 1), tf.int8) # Cast a boolean map into a int map (0/1)
    # Apply crossentropy loss. We use Keras's SparseCategoricalCrossentropy because labels
    # are not one-hot encoded. We let the function transform logits into a probability distribution.
    scceloss = KLS.SparseCategoricalCrossentropy(from_logits=True)
    loss = scceloss(anchor_class, rpn_class_logits)
    # In case the loss tensor is empty (eg. all RPN bboxes were considered neutral), replace it with a 0.
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


@tf.function
def rpn_box_regression_loss_graph(batch_size, target_deltas, rpn_match, rpn_bbox):
    """
    Returns the RPN bounding box loss graph.

    Inputs:
    - batch_size: the batch size of the other tensors 
        (note that this will be converted to a tensor)
    - target_deltas: [batch, max_positive_anchors, (dy, dx, log(dh), log(dw))].
        May be 0-padded in case some bbox deltas are unused.
    - rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    - rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """
    # Like for the classification loss, positive anchors contribute to the loss, 
    # but neutral anchors don't. In addition, also negative anchors don't contribute
    # to the loss, because it's pointless to calculate the deltas of background proposals.
    rpn_match = K.squeeze(rpn_match, -1)
    indices = tf.where(K.equal(rpn_match, 1))
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)

    # We also need to trim target bounding box deltas to the same length
    # 1. Calculate how many positive box are in each image of the batch
    counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
    # For example, counts can be something like [25, 40, 32, 28] for a batch of 4 images.
    # 2. Then, take exactly the number of deltas that have been computed this way
    #    for each slice and concatenate them.
    bbs = tf.zeros([0, 4])
    for i in tf.range(batch_size): # Iterate in the batch
        # This is a workaround to be able to modify the bbs variable inside the loop
        tf.autograph.experimental.set_loop_options(
            shape_invariants=[(bbs, tf.TensorShape([None, 4]))]
        )
        bbs = tf.concat([bbs, tf.slice(target_deltas[i, :, :], 
                                [0,0], [counts[i], 4])], axis=0)    # Start from [0,0] and take the 
                                                                    # appropriate number of boxes.

    # We use the smooth l1 loss, which is more resistent to outliers with respect to
    # classic l1. In TensorFlow, this is implemented as "Huber loss".
    # Basically:
    # loss = 0.5 * x^2                  if |x| <= d
    # loss = 0.5 * d^2 + d * (|x| - d)  if |x| > d
    # Where d is an hyperparameter (default is 1).
    h = KLS.Huber()
    loss = h(bbs, rpn_bbox)
    # As above, in case of empty loss tensor.
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss

#####################################
### MODEL-RELATED UTILS FUNCTIONS ###
#####################################

class NormBoxesLayer(KL.Layer):
    '''
    Custom layer applying normalization to input GT bounding boxes
    '''
    def __init__(self, name, **kwargs):
        super(NormBoxesLayer, self).__init__(name=name, **kwargs)
    
    def call(self, inputs, **kwargs):
        boxes, image = inputs[0], inputs[1]
        shape = tf.shape(image)[1:3] # Ignore batch and other measures
        h, w = tf.split(tf.cast(shape, tf.float32), 2)  # Split in two sub-tensors
        scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)  # Concatenate h and w and reduce them all by 1
        shift = tf.constant([0., 0., 1., 1.])
        return (boxes - shift) / scale

@tf.function
def trim_zeros_tf(boxes, name='trim_zeros'):
    """
    Often boxes are represented as tensors of shape [N, 4] and are padded
    with zeros. This graph function removes zero boxes.

    Inputs:
    - boxes: [N, 4]: matrix of boxes.
    
    Outputs:
    - boxes: [M, 4]: the matrix of boxes cleaned by its zero padded elements
    - non_zeros: [N]: a 1D boolean mask identifying the rows to keep.
    """
    # To create a mask for non-zero elements we can convert non-zero elements
    # to booleans. We know that a 0 is translated to a False, while a non-zero 
    # to True. If the sum of the 4 coordinates of the box is 0, it means that
    # the box is zero-padded, so we can cast it to False.
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros

@tf.function
def calculate_overlaps_matrix_tf(boxes1, boxes2):
    """
    A tensorflow graph function that computes the intersection over union
    metric between two sets of boxes. 

    Inputs:
    - boxes1, boxes2: [N, (y1,x1,y2,x2)]

    Outputs:
    - overlaps: [N,N] matrix containing the IoU metric for each pair of boxes
    """
    # 1. Repeat boxes1 for each element of boxes2. This way we don't need to use
    # loops but we can compare each box of boxes1 with the boxes of boxes2 in parallel!
    b1 = tf.repeat(boxes1, [tf.shape(boxes2)[0]], axis=0) # b1 becomes [N*N, 4]
    # Each row of boxes1 is repeated N times.

    # b2 instead is repeated with the function tile. Similarly to stack, it considers
    # the tensor as a whole block and stacks it on itself N times.
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])

    # The difference is that the first tensor has elements in this way:
    # 111222333
    # While the second tensor has elements like this:
    # 123123123
    # So we can combine them similarly to a loop without doing loops!

    # 2. Compute intersections
    # Get the coordinates from the tensors. Remember that each of these subtensors
    # will be a Nx1 tensor containing coordinates.
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    # Calculate the points of the intersection
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    # Calculate the area intersection. Notice that x2-x1 and y2-y1 can be negative
    # if there is no intersection.
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 4. Compute IoU
    iou = intersection / union
    # 5. Reshape as a NxN matrix.
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps

#####################
##  CODE CEMETERY  ##
#####################

def norm_boxes_tf(boxes, shape):
    '''
    Same as the function in utils_functions with the same name, 
    but using tensorflow to deal with tensors
    '''
    h, w = tf.split(tf.cast(shape, tf.float32), 2)  # Split in two sub-tensors
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)  # Concatenate h and w and reduce them all by 1
    shift = tf.constant([0., 0., 1., 1.])
    return (boxes - shift) / scale

def denorm_boxes_tf(boxes, shape):
    '''
    Same as the function in utils_functions with the same name, 
    but using tensorflow operation to deal with tensors
    '''
    shape = tf.cast(shape, tf.float32)  # Cast the shapes of the image to float32
    h, w = tf.split(shape, 2)  # Split in two sub-tensors
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)  # Concatenate h and w and reduce them all by 1
    shift = tf.constant([0., 0., 1., 1.])
    return tf.cast(tf.round(tf.multiply(boxes, scale) + shift), tf.int32)  # Cast back into pixels