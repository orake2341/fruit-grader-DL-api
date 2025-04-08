import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    DepthwiseConv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Input,
    GlobalAveragePooling2D,
    Reshape,
    Multiply,
    Add,
    Activation,
)
from tensorflow.keras.models import Model
import os


# CBAM: Convolutional Block Attention Module
# Channel Attention
def channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]

    avg_pool = tf.reduce_mean(input_feature, axis=[1, 2], keepdims=True)
    max_pool = tf.reduce_max(input_feature, axis=[1, 2], keepdims=True)

    shared_dense_one = Dense(channel // ratio, activation="relu", use_bias=True)
    shared_dense_two = Dense(channel, use_bias=True)

    avg_out = shared_dense_two(shared_dense_one(avg_pool))
    max_out = shared_dense_two(shared_dense_one(max_pool))

    scale = Activation("sigmoid")(Add()([avg_out, max_out]))
    return Multiply()([input_feature, scale])


# Spatial Attention
def spatial_attention(input_feature):
    avg_pool = tf.reduce_mean(input_feature, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(input_feature, axis=-1, keepdims=True)
    concat = tf.concat([avg_pool, max_pool], axis=-1)
    attention = Conv2D(1, kernel_size=7, padding="same", activation="sigmoid")(concat)
    return Multiply()([input_feature, attention])


# CBAM block
def cbam_block(input_feature):
    x = channel_attention(input_feature)
    x = spatial_attention(x)
    return x


# Define the shared CNN backbone with CBAM
def build_shared_cnn(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(16, (3, 3), activation="relu", padding="valid")(inputs)  # (98,98,16)
    x = DepthwiseConv2D((3, 3), activation="relu", padding="valid")(x)  # (96,96,16)
    x = DepthwiseConv2D((3, 3), activation="relu", padding="valid")(x)  # (94,94,16)
    x = cbam_block(x)  # Apply CBAM attention
    x = MaxPooling2D((3, 3), strides=3, padding="valid")(x)  # (31,31,16)
    x = Flatten()(x)  # (15376,)
    return inputs, x


# Define the multi-task model
def build_multitask_model(input_shape, num_classes):
    inputs, shared_features = build_shared_cnn(input_shape)

    # Task 1: Fruit Freshness (Binary Classification)
    freshness_branch = Dense(54, activation="relu")(shared_features)
    freshness_output = Dense(1, activation="sigmoid", name="freshness_output")(
        freshness_branch
    )

    # Task 2: Fruit Type (Multi-Class Classification)
    type_branch = Dense(54, activation="relu")(shared_features)
    type_output = Dense(num_classes, activation="softmax", name="type_output")(
        type_branch
    )

    # Define the model
    model = Model(inputs=inputs, outputs=[freshness_output, type_output])

    return model


# Load multitask dataset
from tensorflow.keras.utils import image_dataset_from_directory


def extract_labels(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    class_folder = parts[-2]  # e.g., 'freshapple'

    freshness = tf.strings.regex_full_match(class_folder, "fresh.*")
    fruit_type = tf.case(
        [
            (
                tf.strings.regex_full_match(class_folder, ".*apple"),
                lambda: tf.constant(0),
            ),
            (
                tf.strings.regex_full_match(class_folder, ".*banana"),
                lambda: tf.constant(1),
            ),
        ],
        default=lambda: tf.constant(2),
    )

    freshness = tf.cast(freshness, tf.int32)
    return freshness, tf.one_hot(fruit_type, depth=3)


def process_dataset(dataset):
    def map_fn(image, label):
        freshness, fruit = extract_labels(label)
        return image, {"freshness_output": freshness, "type_output": fruit}

    return dataset.map(map_fn)


def load_dataset(directory, batch_size=32):
    dataset = image_dataset_from_directory(
        directory, image_size=(100, 100), label_mode="int", shuffle=True
    )
    return process_dataset(dataset).batch(batch_size).prefetch(tf.data.AUTOTUNE)


# Define model parameters
input_shape = (100, 100, 3)
num_classes = 3  # apple, banana, orange

# Build and compile the model
model = build_multitask_model(input_shape, num_classes)
model.compile(
    optimizer="adam",
    loss={
        "freshness_output": "binary_crossentropy",
        "type_output": "categorical_crossentropy",
    },
    metrics={"freshness_output": "accuracy", "type_output": "accuracy"},
)

# Load datasets
train_dataset = load_dataset("data/train")
valid_dataset = load_dataset("data/valid")
test_dataset = load_dataset("data/test")

# Train the model
model.fit(train_dataset, validation_data=valid_dataset, epochs=10)

# Evaluate the model
model.evaluate(test_dataset)

# Display model summary
model.summary()
