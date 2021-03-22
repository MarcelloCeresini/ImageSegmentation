import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

test_ds, ds_info = tfds.load(
    "mnist",
    split="test",
    as_supervised=True,
    with_info=True
)

print(ds_info)
print(type(test_ds))

def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label


BATCH_SIZE = 128

test_ds = test_ds.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE
)
test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.cache()
test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

# used to retrieve whole model (in case 100 EPOCHS are not enough, you can resume training from this model)
model = tf.keras.models.load_model('saved_model/my_model')

result = model.evaluate(test_ds)

import json
with open('saved_model/my_model/history.txt') as json_file:
    history_dict = json.load(json_file)

val = 0
plot1 = plt.figure(1)
plt.plot(history_dict["loss"][val:])
plt.plot(history_dict["val_loss"][val:])

print(result)
print("a")
