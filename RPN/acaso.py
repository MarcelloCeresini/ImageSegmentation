import tensorflow as tf
import utils_functions as utils

ACCEPTED_CLASSES_IDS = [[1, 2, 3], [4, 7], [9]]
old_new_ids = utils.group_classes(ACCEPTED_CLASSES_IDS)

for a in annotations:
    mask = tf.where(old_new_ids[:,0] == a)
    a = int(tf.gather(old_new_ids[:,1], mask))