import numpy as np

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

def denormalize_image(image, mean_pixel):
    '''
    Takes an image (numpy matrix) in the format expected by the neural network
    (normalized) and transform them back into classical pixel images.

    images: a list of image matrices with different sizes. What is constant
        is the third dimension of the image matrix, the depth (usually 3)
    
    Returns an image (numpy matrix) containing the restored image images ([h, w, 3]).
    '''
    return np.asarray(image + mean_pixel, dtype=np.uint8)

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

