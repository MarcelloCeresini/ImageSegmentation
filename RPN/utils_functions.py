import scipy
import numpy as np
from skimage.transform import resize

from config import ModelConfig

def generate_pyramid_anchors(scales, ratios, feature_shapes, 
                            feature_strides, anchor_stride):
    ''' 
    This function generates anchors at the different levels of the FPN.
    Each scale is bound to a different level of the pyramid
    On the other hand, all ratios of the proposals are used in all levels.

    Returns:
    - anchors: [N, (y1,x1,y2,x2)], an array of all generated anchors.
        Anchors are sorted by scale size.
    '''
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(
            scales[i],              # Use only the appropriate scale for the level
            ratios,                 # Use all ratios for the BBs
            feature_shapes[i],      # At this level, the image has this shape...
            feature_strides[i],     # Or better, is scaled of this quantity
            anchor_stride           # Frequency of sampling in the feature map
                                    # for the generation of anchors
        ))
    # Transform the list in an array [N, (y1,x1,y2,x2)] which contains all generated anchors
    # The sorting of the scale is bound to the scaling of the feature maps,
    # So first we have all anchors at scale 1/4, then all anchors at scale 1/8 and so on...
    return np.concatenate(anchors, axis=0)

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
    widths = scales * np.sqrt(ratios)

    # What are the positions in the feature space?
    # We use arange to create evenly spaced sequences from 0 for all the
    # rows using as a step size the anchor stride.
    # We multiply by the stride of the feature map over the image,
    # so that we get image-aligned coordinates
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
    ).reshape([-1, 2])  # Unstack anchors creating a [total_ancors, 2] array
    box_sizes = np.stack(
        [box_heights, box_widths], axis=2
    ).reshape([-1, 2])

    # Finally, convert the arrays into a single big array with (y1, x1, y2, x2) coordinates
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,  # y1, x1
                            box_centers + 0.5 * box_sizes], axis=1)  # y2, x2
    # Concatenate on the columns obtaining x1,y1,x2,y2
    return boxes


def preprocess_inputs(images, config:ModelConfig):
    '''
    Takes a list of images, modifies them to the format expected as an
    input to the neural network.

    images: a list of image matrices with different sizes. What is constant
        is the third dimension of the image matrix, the depth (usually 3)
    
    Returns a numpy matrix containing the preprocessed image ([N, h, w, 3]).
    The preprocessing includes resizing, zero-padding and normalization.
    '''
    preprocessed_inputs = []
    image_metas = []
    for image in images:
        preprocessed_image, window, scale, padding = resize_image(
            image, config.IMAGE_MIN_DIM, config.IMAGE_MAX_DIM
        )
        # In this case, for "normalize" we mean taking the image,
        # subtracting the mean pixel to it
        # and converting the result to float.
        preprocessed_image = preprocessed_image.astype(np.float32) - config.MEAN_PIXEL
        # Build image meta
        image_meta = compose_image_meta(
            0, # We don't care about the ID in detection mode
            image.shape, preprocessed_image.shape, window, scale
        )
        preprocessed_inputs.append(preprocessed_image)
        image_metas.append(image_meta)
    # Pack into arrays
    preprocessed_inputs = np.stack(preprocessed_inputs)
    image_metas = np.stack(image_metas)
    return preprocessed_inputs, image_metas

def postprocess_detections(rpn_classes, rpn_bboxes, mrcnn_detections, mrcnn_masks, 
                            original_image_shape, preprocessed_image_shape, 
                            window):
    """
    Reformats the detections of one image from the format of the neural
    network output to a format suitable for use in the rest of the
    application.

    Inputs:
    - rpn_classes: [N] classes detected for the RPN bboxes
    - rpn_bboxes: [N, (y1, x1, y2, x2)] in normalized coordinates
    - mrcnn_detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
    - mrcnn_mask: [N, height, width, num_classes]
    - original_image_shape: [H, W, C] Original image shape before resizing
    - preprocessed_image_shape: [H, W, C] Shape of the image after resizing and padding
    - window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real
            image is excluding the padding.

    Returns:
    - rpn_bboxes: [N] RPN classes in pixels
    - rpn_classes: [N, (y1, x1, y2, x2)] RPN boxes in pixels
    - boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
    - class_ids: [N] Integer class IDs for each bounding box
    - scores: [N] Float probability scores of the class_id
    - masks: [height, width, num_instances] Instance masks
    """
    # How many detections do we have?
    # Detections array is padded with zeros. Find the first detection where all
    # coordinates are 0,0,0,0.
    zero_ix = np.where(mrcnn_detections[:, 4] == 0)[0]
    N = zero_ix[0] if zero_ix.shape[0] > 0 else mrcnn_detections.shape[0]

    # Extract boxes, class_ids, scores, and class-specific masks
    boxes = mrcnn_detections[:N, :4]
    class_ids = mrcnn_detections[:N, 4].astype(np.int32)
    scores = mrcnn_detections[:N, 5]
    masks = mrcnn_masks[np.arange(N), :, :, class_ids]

    # Translate normalized coordinates in the resized image to pixel
    # coordinates in the original image before resizing
    norm_window = norm_boxes(window, preprocessed_image_shape[:2])
    wy1, wx1, wy2, wx2 = norm_window
    shift = np.array([wy1, wx1, wy1, wx1])
    wh = wy2 - wy1  # window height
    ww = wx2 - wx1  # window width
    scale = np.array([wh, ww, wh, ww])
    # Convert boxes to normalized coordinates on the window
    rpn_bboxes = np.divide(rpn_bboxes - shift, scale)
    boxes = np.divide(boxes - shift, scale)
    # Convert boxes to pixel coordinates on the original image
    rpn_bboxes = denorm_boxes(rpn_bboxes, original_image_shape[:2])
    boxes = denorm_boxes(boxes, original_image_shape[:2])

    # Filter out detections with zero area. 
    # Happens A LOT in early training when network weights are still random
    exclude_ix = np.where(
        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
    exclude_ix_rpn = np.where(
        (rpn_bboxes[:, 2] - rpn_bboxes[:, 0]) * 
        (rpn_bboxes[:, 3] - rpn_bboxes[:, 1]) <= 0)[0]
    if exclude_ix.shape[0] > 0:
        boxes = np.delete(boxes, exclude_ix, axis=0)
        class_ids = np.delete(class_ids, exclude_ix, axis=0)
        scores = np.delete(scores, exclude_ix, axis=0)
        masks = np.delete(masks, exclude_ix, axis=0)
        N = class_ids.shape[0]
    if exclude_ix_rpn.shape[0] > 0:
        rpn_bboxes = np.delete(rpn_bboxes, exclude_ix_rpn, axis=0)
        rpn_classes = np.delete(rpn_classes, exclude_ix_rpn, axis=0)

    # Resize masks to original image size and set boundary threshold.
    full_masks = []
    for i in range(N):
        # Convert neural network mask to full size mask
        full_mask = postprocess_mask(masks[i], boxes[i], original_image_shape)
        full_masks.append(full_mask)
    full_masks = np.stack(full_masks, axis=-1)\
        if full_masks else np.empty(original_image_shape[:2] + (0,))

    return rpn_bboxes, rpn_classes, boxes, class_ids, scores, full_masks


def postprocess_mask(mask, bbox, image_shape):
    """Converts a mask generated by the neural network to a format similar
    to its original shape. The masks are relative to the bounding boxes, so
    they are applied on only a section of the image.

    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.

    Returns a binary mask with the same size as the original image.
    """
    y1, x1, y2, x2 = bbox
    mask = resize(mask, (y2 - y1, x2 - x1))
    # We want binary mask, so use a threshold to binarize it.
    threshold = 0.5
    mask = np.where(mask >= threshold, 1, 0).astype(np.bool)

    # Apply the mask in the right location.
    full_mask = np.zeros(image_shape[:2], dtype=np.bool)
    full_mask[y1:y2, x1:x2] = mask

    return full_mask


def resize_image(image, min_dim, max_dim):
    """
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
    """
    # Keep track of the image dtype to return the same dtype
    image_dtype = image.dtype
    h, w = image.shape[:2]
    image_max = max(h, w)
    # Scale up, not down
    scale = max(1, min_dim / min(h, w))
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
               (0, 0)]
    # Apply padding to the image
    image = np.pad(image, padding, mode='constant', constant_values=0)
    window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image.astype(image_dtype), window, scale, padding


def resize_mask(mask, scale, padding, crop=None):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    if crop is not None:
        y, x, h, w = crop
        mask = mask[y:y + h, x:x + w]
    else:
        mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask


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


def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


def normalize_image(images, mean_pixel):
    '''
    Takes an image (numpy matrix) in the format read from file and prepares
    it for the neural network by subtracting the mean pixel and converting it
    to float (normalization)

    image: an array of images in RGB format
    
    Returns a float image that is normalized with respect to the mean pixel.
    '''
    return images.astype(np.float32) - mean_pixel


def denormalize_image(image, mean_pixel):
    '''
    Takes an image (numpy matrix) in the format expected by the neural network
    (normalized) and transform them back into classical pixel images.

    image: an image matrix, with depth 3
    
    Returns an image (numpy matrix) containing the restored image images ([h, w, 3]).
    '''
    return np.asarray(image + mean_pixel, dtype=np.uint8)


def compute_overlaps(boxes1, boxes2):
    """
    Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.

    # Prepare matrix
    inters = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    # Iterate over boxes in second set
    for i in range(boxes2.shape[0]):
        y1 = np.maximum(boxes2[i, 0], boxes1[:, 0])
        y2 = np.minimum(boxes2[i, 2], boxes1[:, 2])
        x1 = np.maximum(boxes2[i, 1], boxes1[:, 1])
        x2 = np.minimum(boxes2[i, 3], boxes1[:, 3])
        intersections = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        unions = area2[i] + area1[:] - intersections[:]
        iou = intersections / unions
        # Fill the column that corresponds to box1
        inters[:, i] = iou
    return inters


def log(text:str, array:np.array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  ".format(str(array.shape)))
        if array.size:
            text += ("min: {:10.5f}  max: {:10.5f}".format(array.min(),array.max()))
        else:
            text += ("min: {:10}  max: {:10}".format("",""))
        text += "  {}".format(array.dtype)
    print(text)


def flatten(l:list):
    '''Given a nested list (list of lists) flattens it into one list'''
    return [item for sublist in l for item in sublist]


def group_classes(accepted_ids:list, new_names:list):
    class_ids = []
    for i in range(len(accepted_ids)):
        for element in accepted_ids[i]:
            my_dict = {}
            my_dict["old_id"] = element
            my_dict["new_id"] = i
            my_dict["new_name"] = new_names[i]
            class_ids.append(my_dict)
    return class_ids


def compose_image_meta(image_id, original_image_shape, image_shape,
                       window, scale):
    """Takes attributes of an image and puts them in one 1D array.

    image_id: An int ID of the image. Useful for debugging.
    original_image_shape: [H, W, C] before resizing or padding.
    image_shape: [H, W, C] after resizing and padding
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    scale: The scaling factor applied to the original image (float32)
    """
    meta = np.array(
        [image_id] +                    # size:1
        list(original_image_shape) +    # size:3
        list(image_shape) +             # size:3
        list(window) +                  # size:4 (y1, x1, y2, x2) in image cooredinates
        [scale]                         # size:1
    )
    return meta


def parse_image_meta(meta):
    """Parses an array that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES

    Returns a dict of the parsed values.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    return {
        "image_id": image_id.astype(np.int32),
        "original_image_shape": original_image_shape.astype(np.int32),
        "image_shape": image_shape.astype(np.int32),
        "window": window.astype(np.int32),
        "scale": scale.astype(np.float32)
    }