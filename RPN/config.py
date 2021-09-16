import numpy as np

class ModelConfig():
    # Name for the configuration
    NAME = 'food_detection'

    # Number of GPUs to be used
    GPU_COUNT = 1 # TODO: Only 1 GPU is supported for now

    # Number of images to load on each GPU
    IMAGES_PER_GPU = 1

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

    # Image meta data length
    # See compose_image_meta() for details
    IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1

    ### BACKBONE RELATED CONSTANTS ###

    # Whether to use Resnet50 or Resnet101
    BACKBONE_NETWORK = 'resnet50'  # or 'resnet101'

    # Strides used for computing the shape of each stage of the backbone network
    # (when based on resnet50/101).
    # This is used for aligning proposals to the image. If the image is 1024x1024,
    # the 4 means that the first feature map P2 will be 1024/4 = 256x256. In the 
    # second one we divide by 8 and so on. The last feature map (P6) is 1024/64=16.
    # With these ratio indications we can easily express the relationship between a 
    # feature map and the original image.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    ### FPN RELATED CONSTANTS ###
    # Size of the top-down layers used to build the feature pyramid
    TOP_DOWN_PYRAMID_SIZE = 256

    ### RPN RELATED CONSTANTS ###

    # Scales for the anchor boxes. They represent the square-rooted area
    # of the anchor (so, a scale of 32 means that the anchor has an area of
    # 32^2 pixels). Mathematically, they are sqrt(width*height).
    RPN_ANCHOR_SCALES = [32, 64, 128, 256, 512]

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 2 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1

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
    # You can increase this during training to generate more proposals.
    RPN_NMS_THRESHOLD = 0.7

    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    ### TRAINING RELATED CONSTANTS ###

    # How many training targets for the RPN we want to generate for each image.
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this, so it's reduced to 256.
    # To increase it, we should also decrease the RPN NMS threshold so that
    # more proposals are generated.
    TRAIN_ROIS_PER_IMAGE = 256

    # Percent of positive ROIs used during training to avoid that too many 
    # proposals are used in the loss function.
    ROI_POSITIVE_RATIO = 0.33

    # Number of training steps for each epoch. Defines the number of steps after which
    # we should make a validation step
    STEPS_PER_EPOCH = 1000

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 50

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, decreased to 0.002 after 120k iterations, 
    # but Matterport's implementation reduces this to 0.001 because in their opinion
    # Tensorflow's implementation of the used optimizer causes weights to explode. 
    # For now, we set it at 0.002 following the decreased learning rate from the paper
    # but also staying close to Matterport's implementation.
    LEARNING_RATE = 0.002

    # Momentum is 0.9 from the paper
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization, also taken directly from the paper
    WEIGHT_DECAY = 0.0001

    # Gradient norm clipping. To avoid exploding gradients, the derivatives are clipped
    # to have a L2 norm of maximum 5.0.
    GRADIENT_CLIP_NORM = 5.0

    # Number of classes in the dataset:
    NUM_CLASSES = 28+1    # The dataset has been modified to have 28 classes (+ background class)

    # Accepted classes ids, already in groups
    # Example: ACCEPTED_CLASSES_IDS = [[1040, 1026], [2512, 2504], .....]
    ACCEPTED_CLASSES_IDS = [[1151], [1056], [1154], [1724, 1730, 1728, 1727, 1731], \
        [1566, 1565, 1554, 1607, 1556, 1520, 1538, 1536, 3306, 1561, 1522, 1545, 1557, 1523, 1560, 1551], \
        [2053], [1078], [1311, 1310, 1307, 1352, 1308, 1327, 1280, 1264], [1788, 1789, 1793, 1794], \
        [2131, 2134], [2512, 2504, 2521, 2501, 2524], [1061], [2022], [2099], [1022, 1019], \
        [1505, 1483, 1496, 1506, 1513, 1490, 1487, 1494], [2939, 633, 630], [1010, 1004], \
        [1468, 1469, 1478], [1040, 1026, 1038, 1050], [1967, 2973], \
        [2498, 2530, 2555, 2534, 2543, 2562], [1069], [2738], [2578, 2580, 3080, 3262], [2618], [2620], \
        [5641, 2730, 2711, 2743, 1383, 3332]]

    NEW_NAMES = ["apple", "avocado", "banana", "beef", "bread", "butter", "carrot", "cheese", "chicken", \
        "chocolate", "coffee", "cucumber", "egg", "jam", "mixed-vegetables", "pasta", \
        "pizza", "potatoes", "rice", "salad", "salmon", "tea", "tomato", "tomato-sauce", \
        "water", "red-wine", "white-wine", "yogurt-sauce"]

    # Train or freeze batch normalization layers
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when predicting
    TRAIN_BN = False  # Defaulting to False since batch size is often small

    ### DETECTION HEAD RELATED CONSTANTS ###
    
    # Size of the ROIAlign-pooled ROIs 
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14

    # Layer size inside classification head
    FPN_CLASSIF_LAYERS_SIZE = 1024

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 100

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 100

    ### MASK HEAD RELATED CONSTANTS ###

    # Output mask shape
    # The generated masks are low resolution: 28x28 pixels. 
    # But they are soft masks, represented by float numbers, so they hold more 
    # details than binary masks. The small mask size helps keep the mask 
    # branch light. During training, we scale down the ground-truth masks to 
    # 28x28 to compute the loss, and during inferencing we scale up the predicted 
    # masks to the size of the ROI bounding box and that gives us the final masks, 
    # one per object.
    MASK_SHAPE = [28, 28]  # If we change this, we also need to make adjustments
                                # to the mask branch.

    def __init__(self):
        # Batch size for training and testing
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

    def display(self):
        """Display Configuration values."""
        print()
        print("Configurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print()

