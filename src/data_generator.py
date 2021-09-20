import math
import numpy as np
from tensorflow import keras

import imgaug
import utils_functions as utils

class DataGenerator(keras.utils.Sequence):
    '''
    This class acts as a generator that returns images and corresponding target class ids,
    bounding box deltas, and masks.
    We use a generator for efficiency so that we don't have to keep in memory
    the whole dataset of images and annotations, but we can apply arbitrary 
    transformations on them on the fly, when requested.
    We inherit from keras.utils.Sequence to have some nice properties like
    multiprocessing support.

    The goal of this class is to implement a flexible data generator.
    Upon calling next() on the generator, a batch of two lists are returned: 
    one for inputs to the model and the other for outputs. The contents
    of the lists differ depending on the received arguments:

    - inputs list:
        - images: [batch, H, W, C]
        - image_meta: [batch, shape] Original shapes for the image.
        - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
        - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
        - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
        - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
        - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
                    are those of the image.

    - outputs list: Usually empty in regular training. But if detection_targets
        is True then the outputs list contains target class_ids, bbox deltas,
        and masks.
    '''
    def __init__(self, dataset, config, 
                        shuffle=True, augmentation=None, 
                        detection_targets=False, dont_normalize=False):
        '''
        Inputs:
        - dataset: The dataset to pick data from
        - config: The ModelConfig object containing all configuration for training
        - shuffle: Whether to shuffle samples before every epoch
        - augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
            For example, passing imgaug.augmenters.Fliplr(0.5) flips images
            right/left 50% of the time.
        - detection_targets: If True, generate detection targets (class IDs, bbox
            deltas, and masks). Typically for debugging or visualizations because
            in training detection targets are generated by DetectionTargetLayer.
        - dont_normalize: Do not normalize the image (for visualization, for example)
        '''
        super().__init__()
        self.dataset = dataset
        self.config = config
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.detection_targets = detection_targets
        self.dont_normalize = dont_normalize
        
        self.image_ids = np.copy(dataset.image_ids)
        self.error_count = 0

        self.backbone_shapes = np.array(
        [
            [int(math.ceil(config.IMAGE_SHAPE[0] / stride)),
            int(math.ceil(config.IMAGE_SHAPE[1] / stride))]
        for stride in config.BACKBONE_STRIDES])

        self.anchors = utils.generate_pyramid_anchors(
            config.RPN_ANCHOR_SCALES,
            config.RPN_ANCHOR_RATIOS,
            self.backbone_shapes,
            config.BACKBONE_STRIDES,
            config.RPN_ANCHOR_STRIDE
        )

        # Flatten the list of accepted classes. 
        # Save the result, so we only compute it once.
        # TODO: read below: we may not need this.
        # self.accepted_classes = utils.flatten(self.config.ACCEPTED_CLASSES_IDS)

        self.on_epoch_end()

    def __getitem__(self, index):
        b = 0 # Batch item index: this function returns a whole batch
        while b < self.config.BATCH_SIZE:
            try: 
                # Get GT bounding box and masks for image
                try:
                    image_id = self.image_ids[index + b]
                except IndexError:
                    # In case we choose the last index
                    image_id = self.image_ids[index - b]

                image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                    self.load_image_gts(image_id) 

                # ##########################
                # TODO: Techincally when the dataset in food.py, only images
                # who have at least an element of one of the loaded classes match, so it
                # shouldn't be necessary to add this test here.
                # Anyway, we can't do it this way, because we could potentially 
                # yield empty batches.
                # Skip images that have no instances. This can happen in cases
                # where we train on a subset of classes and the image doesn't
                # have any of the classes we care about.
                #if not np.any(gt_class_ids > 0):
                #    continue
                # We only want images with accepted classes, we do not keep all classes
                # So we also skip images that do not contain any annotation that we accepted
                # "set" is used because intersection is a method of the set class
                # "flatten" takes a nested list and flattens it
                #intersection = set(gt_class_ids).intersection(self.accepted_classes)
                #if intersection == set():
                #    continue
                ###########################
            
                # RPN targets
                rpn_match, rpn_bbox = self.build_rpn_targets(self.anchors,
                                    gt_class_ids, gt_boxes, self.config)

                if b == 0:
                    # Init batch arrays. We are doing it here because we need some of the
                    # previously computed variables.
                    batch_image_meta = np.zeros((self.config.BATCH_SIZE,) + image_meta.shape,
                        dtype=image_meta.dtype)
                    batch_rpn_match = np.zeros(
                        [self.config.BATCH_SIZE, self.anchors.shape[0], 1], dtype=rpn_match.dtype)
                    batch_rpn_bbox = np.zeros(
                        [self.config.BATCH_SIZE, self.config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], 
                            dtype=rpn_bbox.dtype)
                    batch_images = np.zeros(
                        (self.config.BATCH_SIZE,) + image.shape, dtype=np.float32)
                    batch_gt_class_ids = np.zeros(
                        (self.config.BATCH_SIZE, self.config.MAX_GT_INSTANCES), dtype=np.int32)
                    batch_gt_boxes = np.zeros(
                        (self.config.BATCH_SIZE, self.config.MAX_GT_INSTANCES, 4), dtype=np.int32)
                    batch_gt_masks = np.zeros(
                        (self.config.BATCH_SIZE, gt_masks.shape[0], gt_masks.shape[1],
                            self.config.MAX_GT_INSTANCES), dtype=gt_masks.dtype)
                    
                # If more GT boxes than the ones we have prepared in the array, sub-sample from them.
                if gt_boxes.shape[0] > self.config.MAX_GT_INSTANCES:
                    ids = np.random.choice(np.arange(gt_boxes.shape[0]), 
                                self.config.MAX_GT_INSTANCES, 
                                replace=False)
                    gt_class_ids = gt_class_ids[ids]
                    gt_boxes = gt_boxes[ids]
                    gt_masks = gt_masks[:, :, ids]

                # Add to batch
                batch_image_meta[b] = image_meta
                batch_rpn_match[b] = rpn_match[:, np.newaxis]
                batch_rpn_bbox[b] = rpn_bbox
                batch_images[b] = utils.normalize_image(image.astype(np.float32), self.config.MEAN_PIXEL) if \
                                    not self.dont_normalize else image
                batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
                batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
                batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks

                b += 1
            except:
                print("Error processing image with index {}".format(index))
                self.error_count += 1
                if self.error_count > 5:
                    # Too many errors, stop early and see what's wrong.
                    raise
        
        # We have a full batch. It's time to return the generated data!
        inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                    batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
        outputs = []

        return inputs, outputs

    def load_image_gts(self, id):
        '''
        Load and return ground truth data for an image (image, mask, bounding boxes).

        Inputs:
            - id: The ID of the image

        Returns:
            - image: [height, width, 3]
            - shape: the original shape of the image before resizing and cropping.
            - class_ids: [instance_count] Integer class IDs
            - bbox: [instance_count, (y1, x1, y2, x2)]
            - mask: [height, width, instance_count]. The height and width are those
                of the image.
        '''
        image = self.dataset.load_image(id)
        mask, class_ids = self.dataset.load_mask(id)
        original_shape = image.shape
        image, window, scale, padding = utils.resize_image(
            image,
            min_dim = self.config.IMAGE_MIN_DIM,
            max_dim = self.config.IMAGE_MAX_DIM
        )
        mask = utils.resize_mask(mask, scale, padding)

        if self.augmentation is not None:
            # Augmenters that are safe to apply to masks
            # Some, such as Affine, have settings that make them unsafe, so always
            # test your augmentation on masks
            MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                                "Fliplr", "Flipud", "CropAndPad",
                                "Affine", "PiecewiseAffine"]

            def hook(images, augmenter, parents, default):
                """Determines which augmenters to apply to masks."""
                return augmenter.__class__.__name__ in MASK_AUGMENTERS

            # Store shapes before augmentation to compare
            image_shape = image.shape
            mask_shape = mask.shape
            # Make augmenters deterministic to apply similarly to images and masks
            det = self.augmentation.to_deterministic()
            image = det.augment_image(image)
            # Change mask to np.uint8 because imgaug doesn't support np.bool
            mask = det.augment_image(mask.astype(np.uint8),
                                    hooks=imgaug.HooksImages(activator=hook))
            # Verify that shapes didn't change
            assert image.shape == image_shape, "Augmentation shouldn't change image size"
            assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
            # Change mask back to bool
            mask = mask.astype(np.bool)
            
        # Note that some boxes might be all zeros if the corresponding mask got cropped out.
        # so we filter them out here
        _idx = np.sum(mask, axis=(0, 1)) > 0
        mask = mask[:, :, _idx]
        class_ids = class_ids[_idx]
        # Bounding boxes. Note that some boxes might be all zeros
        # if the corresponding mask got cropped out.
        # bbox: [num_instances, (y1, x1, y2, x2)]
        bbox = utils.extract_bboxes(mask)

        # Collect metadata about the image in an array
        image_meta = utils.compose_image_meta(id, original_shape, 
                                        image.shape, window, scale)

        return image, image_meta, class_ids, bbox, mask

    def build_rpn_targets(self, anchors, gt_class_ids, 
                            gt_boxes, config):
        """Given an array containing the anchors and an array containing GT boxes, 
        compute overlaps in order to match anchors to GT boxes, identify positive
        anchors and distinguish them from negatives and compute deltas to refine 
        them to match their corresponding GT boxes.

        anchors: [num_anchors, (y1, x1, y2, x2)]
        gt_class_ids: [num_gt_boxes] Integer class IDs.
        gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

        Returns:
        rpn_match: [N] (int32) matches between anchors and GT boxes.
                1 = positive anchor, -1 = negative anchor, 0 = neutral
        rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
        """

        # Prepare RPN Match array:   1 = positive anchor, 
        #                           -1 = negative anchor, 
        #                            0 = neutral
        # Remember: 
        # Positive = IoU > 0.7 or the highest between IoUs for a given GT box
        # Negative = IoU < 0.3
        # Neutral  = not positive nor negative: they do not contribute to the loss
        rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
        # Prepare the RPN bounding boxes array: [max anchors per image, (dy, dx, log(dh), log(dw))]
        rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

        # Handle COCO crowds
        # A crowd box in COCO is a bounding box around several instances. Exclude
        # them from training. A crowd box is given a negative class ID.
        # This is later handled by the model's anchors refinement mechanism.
        crowd_ix = np.where(gt_class_ids < 0)[0]
        if crowd_ix.shape[0] > 0:
            # Filter out crowds from ground truth class IDs and boxes
            non_crowd_ix = np.where(gt_class_ids > 0)[0]
            crowd_boxes = gt_boxes[crowd_ix]
            gt_class_ids = gt_class_ids[non_crowd_ix]
            gt_boxes = gt_boxes[non_crowd_ix]
            # Compute overlaps with crowd boxes [anchors, crowds]
            crowd_overlaps = utils.compute_overlaps(anchors, crowd_boxes)
            crowd_iou_max = np.amax(crowd_overlaps, axis=1)
            # Create a mask telling which anchors intersect a crowd GT.
            no_crowd_bool = (crowd_iou_max < 0.001)
        else:
            # All anchors don't intersect a crowd
            no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

        # Compute overlaps. We obtain a matrix with the anchors on the rows
        # and the GT boxes on the columns where each value corresponds to the IoU
        # between the selected anchor and GT box.
        overlaps = utils.compute_overlaps(anchors, gt_boxes)

        # Match anchors to GT Boxes
        # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
        #   Also, the anchor which has the highest IoU with a GT box is assigned a
        #   positive class even when it's IoU is < 0.7. This means that there is at
        #   least a positive match for each GT box.
        # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
        #   Crowd-matched anchors are also assigned a negative class.
        # Neutral anchors are those that don't match the conditions above,
        # and they don't influence the loss function.
        # However, don't keep any GT box unmatched (rare, but happens). Instead,
        # match it to the closest anchor (even if its max IoU is < 0.3).

        # 1. Set negative anchors first. They get overwritten below if a GT box is
        # matched to them. Skip boxes in GT crowd areas.
        # Get the index of the highest scoring anchors (axis=1 is rows)
        anchor_iou_argmax = np.argmax(overlaps, axis=1) 
        # Get the highest IoUs for each anchor.
        anchor_iou_max = np.max(overlaps, axis=1) 
        # Assign negative classes to crowd and max IoU < 0.3 anchors
        rpn_match[np.logical_and(anchor_iou_max < 0.3, no_crowd_bool)] = -1

        # 2. Set an anchor for each GT box (regardless of IoU value).
        # If multiple anchors have the same IoU, match all of them
        # Operationally:
        # Take the indexes of the anchors where the IoU matches 
        # the max ones over the columns (axis=0)
        gt_iou_argmax = np.argwhere(overlaps == np.max(overlaps, axis=0))[:,0] 
        rpn_match[gt_iou_argmax] = 1

        # 3. Set anchors with high overlap as positive.
        rpn_match[anchor_iou_max >= 0.7] = 1

        # Subsample to balance positive and negative anchors
        # Don't let positives be more than half the anchors
        ids = np.where(rpn_match == 1)[0]
        extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
        if extra > 0:
            # Reset the extra ones to neutral
            ids = np.random.choice(ids, extra, replace=False)
            rpn_match[ids] = 0
        # Same for negative proposals
        ids = np.where(rpn_match == -1)[0]
        extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                            np.sum(rpn_match == 1))
        if extra > 0:
            # Rest the extra ones to neutral
            ids = np.random.choice(ids, extra, replace=False)
            rpn_match[ids] = 0

        # For positive anchors, compute shift and scale needed to transform them
        # to match the corresponding GT boxes.
        ids = np.where(rpn_match == 1)[0]
        ix = 0  # index into rpn_bbox
        for i in ids:
            a = anchors[i]
            # Closest gt box (it might have IoU < 0.7)
            gt = gt_boxes[anchor_iou_argmax[i]]

            # Convert coordinates to center plus width/height.
            # GT Box
            gt_h = gt[2] - gt[0]
            gt_w = gt[3] - gt[1]
            gt_center_y = gt[0] + 0.5 * gt_h
            gt_center_x = gt[1] + 0.5 * gt_w
            # Anchor
            a_h = a[2] - a[0]
            a_w = a[3] - a[1]
            a_center_y = a[0] + 0.5 * a_h
            a_center_x = a[1] + 0.5 * a_w

            # Compute the bbox refinement that the RPN should predict.
            rpn_bbox[ix] = [
                (gt_center_y - a_center_y) / a_h,
                (gt_center_x - a_center_x) / a_w,
                np.log(gt_h / a_h),
                np.log(gt_w / a_w),
            ]
            # Normalize
            # TODO: We are not doing this in our code (should we?)
            # rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
            ix += 1

        return rpn_match, rpn_bbox

    def on_epoch_end(self):
        super().on_epoch_end()
        if self.shuffle:
            np.random.shuffle(self.image_ids)

    def __len__(self):
        '''
        Denotes how many batches we have for each epoch. We can call the Generator
        as many times as this len function returns: after that the epoch is considered
        finished, the on_epoch_end function is called and a new epoch starts.
        '''
        return int(np.floor(len(self.dataset.image_ids) / self.config.BATCH_SIZE))

    @property
    def iterator(self):
        return self.__iter__()