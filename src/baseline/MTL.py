import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    DepthwiseConv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import image_dataset_from_directory
import os


# Define the shared CNN backbone
def build_shared_cnn(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(16, (3, 3), activation="relu", padding="valid")(inputs)  # (98,98,16)
    x = DepthwiseConv2D((3, 3), activation="relu", padding="valid")(x)  # (96,96,16)
    x = DepthwiseConv2D((3, 3), activation="relu", padding="valid")(x)  # (94,94,16)
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


# Extract custom labels from integer labels and class names
def extract_labels_from_classname(class_name):
    freshness = tf.cast(tf.strings.regex_full_match(class_name, "fresh.*"), tf.int32)

    fruit_type = tf.case(
        [
            (
                tf.strings.regex_full_match(class_name, ".*apple"),
                lambda: tf.constant(0),
            ),
            (
                tf.strings.regex_full_match(class_name, ".*banana"),
                lambda: tf.constant(1),
            ),
        ],
        default=lambda: tf.constant(2),
    )
    one_hot_fruit = tf.one_hot(fruit_type, depth=3)
    return freshness, one_hot_fruit


def process_dataset(dataset, class_names, batch_size):
    def map_fn(image, label):
        class_name = tf.gather(class_names, label)
        freshness, fruit = extract_labels_from_classname(class_name)
        return image, {
            "freshness_output": tf.cast(freshness, tf.float32),
            "type_output": fruit,
        }

    return (
        dataset.unbatch()
        .map(map_fn)
        .repeat()
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def load_dataset(directory, batch_size=8):
    dataset = image_dataset_from_directory(
        directory,
        image_size=(100, 100),
        label_mode="int",
        batch_size=batch_size,
        shuffle=True,
    )
    class_names = tf.constant(dataset.class_names)
    return process_dataset(dataset, class_names, batch_size).prefetch(tf.data.AUTOTUNE)


# Define model parameters
input_shape = (100, 100, 3)
num_classes = 3

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
train_dataset = load_dataset("../../data/newDataset/train")
valid_dataset = load_dataset("../../data/newDataset/valid")
test_dataset = load_dataset("../../data/newDataset/test")

# Train the model
model.fit(train_dataset, validation_data=valid_dataset, epochs=15)

# Evaluate the model
model.evaluate(test_dataset)

# Display model summary
model.summary()
