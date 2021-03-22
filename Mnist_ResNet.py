import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

(train_ds, test_ds), ds_info = tfds.load(
    "mnist",
    split=["train[:80%]", "test"],
    as_supervised=True,
    with_info=True
)

val_ds, ds_info_val = tfds.load(
    "mnist",
    split="train[80%:]",
    as_supervised=True,
    with_info=True
)


BATCH_SIZE = 128
input_shape = ds_info.features["image"].shape
num_classes = ds_info.features["label"].num_classes
num_examples_train = int(ds_info.splits["train"].num_examples * 0.8)
num_examples_val = int(num_examples_train / 0.8 * 0.2)


# num_examples_test = ds_info.splits["test"].num_examples


def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label


train_ds = train_ds.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE
)
train_ds = train_ds.cache()
train_ds = train_ds.shuffle(num_examples_train)
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

val_ds = val_ds.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE
)
val_ds = val_ds.cache()
val_ds = val_ds.shuffle(num_examples_val)
val_ds = val_ds.batch(BATCH_SIZE)
val_ds = val_ds.prefetch(tf.data.experimental.AUTOTUNE)

test_ds = test_ds.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE
)
test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.cache()
test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)


def res_identity(x, filters):  # dimension does not change
    x_skip = x
    f1, f2 = filters

    # first block
    x = layers.Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding="valid", kernel_regularizer=l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(relu)(x)

    # second block, BOTTLENECK (but size kept SAME WITH PADDING!)
    x = layers.Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_regularizer=l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(relu)(x)

    # third block + input
    x = layers.Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding="valid", kernel_regularizer=l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, x_skip])
    x = layers.Activation(relu)(x)

    return x


def res_conv(x, s, filters):  # dimension changes, divided by s
    x_skip = x
    f1, f2 = filters

    # first block
    x = layers.Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding="valid", kernel_regularizer=l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(relu)(x)

    # second block, bottleneck, "same" padding
    x = layers.Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_regularizer=l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(relu)(x)

    # third block
    x = layers.Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding="valid", kernel_regularizer=l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x_skip = layers.Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding="valid", kernel_regularizer=l2(0.001))(
        x_skip)
    x_skip = layers.BatchNormalization()(x_skip)
    x = layers.Add()([x, x_skip])
    x = layers.Activation(relu)(x)

    return x


def resnet():
    inputs = layers.Input(shape=input_shape)
    x = layers.ZeroPadding2D(padding=(3, 3))(inputs)

    # 1st stage --> maxpooling
    x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(relu)(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # 2nd stage
    x = res_conv(x, s=1, filters=(64, 256))
    x = res_identity(x, filters=(64, 256))
    # x = res_identity(x, filters=(64, 256))

    # 3rd stage
    x = res_conv(x, s=2, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))
    # x = res_identity(x, filters=(128, 512))

    # 4th stage
    x = res_conv(x, s=2, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    # x = res_identity(x, filters=(256, 1024))
    # x = res_identity(x, filters=(256, 1024))
    # x = res_identity(x, filters=(256, 1024))
    # x = res_identity(x, filters=(256, 1024))

    # 5th stage
    x = x = res_conv(x, s=2, filters=(512, 2048))
    x = res_identity(x, filters=(512, 2048))
    # x = res_identity(x, filters=(512, 2048))

    # avg pooling + dense
    x = layers.AveragePooling2D(pool_size=(2, 2), padding="same")(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation="softmax", kernel_initializer="he_normal")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


model = resnet()
model.summary()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics="accuracy"
)

checkpoint_path = "training_1/cp.{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
save_frequency = 1

model.save_weights(checkpoint_path.format(epoch=0))

'''
# used to retrieve the checkpoints
latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)
'''

EPOCHS = 50
history = model.fit(
    train_ds,
    validation_data=val_ds,
    batch_size=BATCH_SIZE,
    validation_batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                           save_weights_only=True,
                                           verbose=1,
                                           save_freq=save_frequency * BATCH_SIZE)
    ]
)

val = 0
plot1 = plt.figure(1)
plt.plot(history.history["loss"][val:])
plt.plot(history.history["val_loss"][val:])

os.mkdir("saved_model")
model.save('saved_model/my_model')

'''
# used to retrieve whole model (in case 100 EPOCHS are not enough, you can resume training from this model)
new_model = tf.keras.models.load_model('saved_model/my_model')
'''
