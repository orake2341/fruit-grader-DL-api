import os

os.environ["KERAS_BACKEND"] = "torch"
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from random import shuffle
import pickle
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import keras_core as keras
from keras_core.models import Model, Sequential, load_model
from keras_core.layers import (
    Dense,
    Dropout,
    Conv2D,
    MaxPooling2D,
    Activation,
    Flatten,
    BatchNormalization,
    SeparableConv2D,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Reshape,
    Multiply,
    Add,
    Concatenate,
    Lambda,
)
from keras_core import ops
from keras_core import Input
from keras_core.utils import to_categorical, plot_model
import torch
from random import random
import pickle
from IPython.core.display import display, HTML
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras_core.applications import EfficientNetB3
import keras_core
from keras_core import Layer
from keras_core.saving import register_keras_serializable

keras_core.config.enable_unsafe_deserialization()
# switch to torch backend


torch.cuda.is_available()
print("CUDA Available:", torch.cuda.is_available())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

DATA_PATH = "../../data/Dataset"
PICKLE_FILE = "../../data/fruit_mtl_data.pkl"


def load_rand():
    X = []
    for sub_dir in os.listdir(DATA_PATH):
        print(sub_dir)
        path_main = os.path.join(DATA_PATH, sub_dir)
        i = 0
        for img_name in os.listdir(path_main):
            if i >= 6:
                break
            img = cv2.imread(os.path.join(path_main, img_name))
            img = cv2.resize(img, (100, 100))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X.append(img)
            i += 1
    return X


X = load_rand()

X = np.array(X)
X.shape


def show_subpot(X, title=False, Y=None):
    total_images = X.shape[0]
    rows = 6  # Define the number of rows you want
    cols = total_images // rows  # Automatically adjust the columns

    f, ax = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    for i, img in enumerate(X):
        row, col = divmod(i, cols)  # Calculate the row and column for each subplot
        ax[row, col].imshow(img, aspect="auto")
        if title:
            ax[row, col].set_title(Y[i] if Y is not None else str(i))

    plt.show()


show_subpot(X)
del X


def load_mtl_data(path=DATA_PATH):
    quality = ["Fresh", "Rotten"]
    cat = [
        "Apple",
        "Banana",
        "Grape",
        "Guava",
        "Jujube",
        "Orange",
        "Pomegranate",
        "Strawberry",
    ]
    X, Y = [], []
    z = []
    for cata in tqdm(os.listdir(path)):
        path_main = os.path.join(path, cata)

        for i, name in enumerate(quality):
            if quality[i] in cata:
                fresh_index = i
                print(f"quality Index: {fresh_index}")
                print(f"quality Index: {name}")
                break

        for i, name in enumerate(cat):
            if cat[i] in cata:
                cat_index = i
                print(f"Category Index: {cat_index}")
                print(f"Category Index: {name}")
                break

        for img_name in os.listdir(path_main):
            img = cv2.imread(os.path.join(path_main, img_name))
            img = cv2.resize(img, (100, 100))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            z.append([img, [fresh_index, cat_index]])

    print("Loading and shuffling image data ... ...")
    shuffle(z)
    for images, labels in z:
        X.append(images)
        Y.append(labels)

    X = np.array(X) / 255.0

    label1 = []
    label2 = []

    for label in Y:
        label1.append(label[0])
        label2.append(to_categorical(label[1], num_classes=8))

    Y = [np.array(label1), np.array(label2)]

    return X, Y  # , X_cal, y_cal, X_test, y_test


if not os.path.exists(PICKLE_FILE):
    X, Y = load_mtl_data(path=DATA_PATH)

    label_all = []
    for i in range(len(X)):
        label_all.append(str(Y[0][i]) + str(np.argmax(Y[1][i])))
    label_all = np.array(label_all)
    pickle.dump([X, Y, label_all], open(PICKLE_FILE, "wb"))

else:
    X, Y, label_all = pickle.load(open(PICKLE_FILE, "rb"))
    print("X shape from pickle:", X.shape)


def show_images(images, labels, n_images=5):
    """
    Display a grid of images with their corresponding labels.
    Args:
        images (numpy.ndarray): The images to display.
        labels (numpy.ndarray): The corresponding labels for the images.
        n_images (int): The number of images to display.
    """
    fig, axes = plt.subplots(1, n_images, figsize=(15, 5))
    for i in range(n_images):
        ax = axes[i]
        ax.imshow((images[i]))
        ax.axis("off")
        label = labels[i]
        ax.set_title(f"Label: {label}")
    plt.show()


def split_data(test_size=0.3, seed=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, label_all, test_size=test_size, stratify=label_all, random_state=seed
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.3, stratify=y_train, random_state=seed
    )
    print(f"X_train dtype: {X_train.dtype}, min: {X_train.min()}, max: {X_train.max()}")
    show_images(X_train, y_train, n_images=5)

    def split_labels(label_comb):
        y1 = []  # Freshness
        y2 = []  # Category (one-hot)
        for item in label_comb:
            fresh = int(item[0])
            cat = int(item[1])
            y1.append(fresh)
            one_hot = [0] * 8
            one_hot[cat] = 1
            y2.append(one_hot)
        return np.array(y1), np.array(y2)

    print(f"X_train dtype: {X_val.dtype}, min: {X_val.min()}, max: {X_val.max()}")
    print(f"X_train dtype: {X_test.dtype}, min: {X_test.min()}, max: {X_test.max()}")
    y_train_t1, y_train_t2 = split_labels(y_train)
    y_val_t1, y_val_t2 = split_labels(y_val)
    y_test_t1, y_test_t2 = split_labels(y_test)

    print(f"testT1:{y_test_t1}")
    print(f"testT2:{y_test_t2}")

    print(f"validT1:{y_val_t1}")
    print(f"validT2:{y_val_t2}")
    y_val_t1 = np.array(y_val_t1)
    y_val_t2 = np.array(y_val_t2)

    y_train_mtl = [y_train_t1, y_train_t2]
    y_test_mtl = [y_test_t1, y_test_t2]
    y_val_mtl = [y_val_t1, y_val_t2]

    return (
        X_train,
        X_test,
        y_test_t1,
        y_test_t2,
        y_train_mtl,
        y_test_mtl,
        X_val,
        y_val_mtl,
    )


# global settings

RUNS = 1  # the full experiment runs 50 times.

K_NUM = 32  # kernel num, e.g., 64
EMBEDDING_DIM = 64  # the final embedding size, e.g., 128

EPOCHS = 15  # max epochs to train for various models
BATCH_SIZE = 16  # 18 default


def base_net():
    base_model = Sequential()
    base_model.add(Conv2D(K_NUM, (3, 3), activation="relu", input_shape=(100, 100, 3)))
    base_model.add(
        SeparableConv2D(
            K_NUM,
            (3, 3),
            depthwise_initializer="he_uniform",
            pointwise_initializer="he_uniform",
            activation="relu",
        )
    )
    base_model.add(
        SeparableConv2D(
            K_NUM,
            (3, 3),
            depthwise_initializer="he_uniform",
            pointwise_initializer="he_uniform",
            activation="relu",
        )
    )
    base_model.add(MaxPooling2D((3, 3)))
    base_model.add(Flatten())
    return base_model


# def channel_attention(input_feature, ratio=4):
#     channel = input_feature.shape[-1]

#     avg_pool = GlobalAveragePooling2D()(input_feature)
#     max_pool = GlobalMaxPooling2D()(input_feature)

#     shared_dense_1 = Dense(channel // ratio, activation="relu", use_bias=True)
#     shared_dense_2 = Dense(channel, use_bias=True)

#     avg_out = shared_dense_1(avg_pool)
#     avg_out = shared_dense_2(avg_out)

#     max_out = shared_dense_1(max_pool)
#     max_out = shared_dense_2(max_out)

#     cbam_feature = Add()([avg_out, max_out])
#     cbam_feature = Activation("sigmoid")(cbam_feature)
#     cbam_feature = Reshape((1, 1, channel))(cbam_feature)

#     return Multiply()([input_feature, cbam_feature])


# def spatial_attention(input_feature):
#     avg_pool = ops.mean(input_feature, axis=3, keepdims=True)
#     max_pool = ops.max(input_feature, axis=3, keepdims=True)

#     concat = Concatenate(axis=-1)([avg_pool, max_pool])
#     cbam_feature = Conv2D(
#         filters=1, kernel_size=7, strides=1, padding="same", activation="sigmoid"
#     )(concat)
#     return Multiply()([input_feature, cbam_feature])


# def cbam_block(input_feature, ratio=4):
#     feature = channel_attention(input_feature, ratio)
#     feature = spatial_attention(feature)
#     return feature


@register_keras_serializable()
class CBAMBlock(Layer):
    def __init__(self, ratio=4, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        self.channel = input_shape[-1]

        # Channel attention layers
        self.shared_dense_1 = Dense(
            self.channel // self.ratio, activation="relu", use_bias=True
        )
        self.shared_dense_2 = Dense(self.channel, use_bias=True)
        self.reshape = Reshape((1, 1, self.channel))

        # Spatial attention layers
        self.concat = Concatenate(axis=-1)
        self.conv2d = Conv2D(
            filters=1, kernel_size=7, strides=1, padding="same", activation="sigmoid"
        )

    def call(self, input_feature):
        # Channel attention
        avg_pool = GlobalAveragePooling2D()(input_feature)
        max_pool = GlobalMaxPooling2D()(input_feature)

        avg_out = self.shared_dense_2(self.shared_dense_1(avg_pool))
        max_out = self.shared_dense_2(self.shared_dense_1(max_pool))
        cbam_feature = Activation("sigmoid")(Add()([avg_out, max_out]))
        cbam_feature = self.reshape(cbam_feature)
        channel_refined = Multiply()([input_feature, cbam_feature])

        # Spatial attention
        avg_pool = ops.mean(channel_refined, axis=3, keepdims=True)
        max_pool = ops.max(channel_refined, axis=3, keepdims=True)
        concat = self.concat([avg_pool, max_pool])
        cbam_feature = self.conv2d(concat)
        refined_feature = Multiply()([channel_refined, cbam_feature])

        return refined_feature


def base_net_cbam():
    inputs = Input(shape=(100, 100, 3))

    x = Conv2D(K_NUM, (3, 3), padding="same", activation="relu")(inputs)
    x = CBAMBlock(ratio=4)(x)
    x = SeparableConv2D(
        K_NUM,
        (3, 3),
        padding="same",
        depthwise_initializer="he_uniform",
        pointwise_initializer="he_uniform",
        activation="relu",
    )(x)
    x = CBAMBlock(ratio=4)(x)
    x = SeparableConv2D(
        K_NUM,
        (3, 3),
        padding="same",
        depthwise_initializer="he_uniform",
        pointwise_initializer="he_uniform",
        activation="relu",
    )(x)
    x = MaxPooling2D((3, 3))(x)
    x = CBAMBlock(ratio=4)(x)
    x = Flatten()(x)

    return Model(inputs, x)


def show_feature_maps(model, X_test, layer_name, n_images=5):
    try:
        # Try getting "sequential" first
        backbone_layer = model.get_layer("sequential")
    except ValueError:
        # If not found, fall back to "functional"
        backbone_layer = model.get_layer("functional_15")

    # Step 2: Create a model up to the target layer inside sequential
    inputs = keras_core.Input(shape=model.input_shape[1:])  # fresh input
    x = inputs

    def apply_layers(layers, x, target_layer_name):
        for layer in layers:
            # Always pass training=False
            try:
                x_out = layer(x, training=False)
            except Exception as e:
                print(f"Skipping layer {layer.name} due to error: {e}")
                x_out = x  # if layer fails, just continue

            if layer.name == target_layer_name:
                return x_out, True  # found target layer

            # If layer is a model (like CBAM), go deeper
            if isinstance(layer, Model):
                x_out, found = apply_layers(layer.layers, x_out, target_layer_name)
                if found:
                    return x_out, True

            x = x_out  # update x to continue
        return x, False  # not found

    # Apply layers until target layer
    x, _ = apply_layers(backbone_layer.layers, x, layer_name)

    feature_model = Model(inputs=inputs, outputs=x)  # build small feature map model
    print(feature_model.summary(expand_nested=True))

    # Step 3: Predict
    feature_maps = feature_model.predict(X_test[:n_images])

    # Step 4: Plot
    for i in range(n_images):
        plt.figure(figsize=(15, 15))
        n_features = feature_maps.shape[-1]
        size = int(np.ceil(np.sqrt(n_features)))

        for j in range(n_features):
            plt.subplot(size, size, j + 1)
            plt.imshow(feature_maps[i, :, :, j], cmap="viridis")
            plt.axis("off")

        plt.show()


def draw_training_curve(history, title=""):
    plt.figure(1, figsize=(20, 8))
    # plt.title('STL' + title)

    plt.subplot(1, 2, 1)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.grid(True)
    plt.legend()
    plt.show()


def draw_training_curve_MTL(history, architecture=None):
    plt.figure(1, figsize=(20, 8))
    filename = f"{architecture}_training_curve_MTL.png"
    plt.subplot(1, 3, 1)
    plt.xlabel("Epochs")
    plt.ylabel("Total Loss")
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.xlabel("Epochs")
    plt.ylabel("T1 Accuracy")
    plt.plot(history.history["fresh_accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_fresh_accuracy"], label="Validation Accuracy")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.xlabel("Epochs")
    plt.ylabel("T2 Accuracy")
    plt.plot(history.history["cat_accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_cat_accuracy"], label="Validation Accuracy")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)  # Save the plot to file
    plt.show()


def evaluate_t1(
    X_test,
    y_test,
    best_model,
    MTL=False,
    show_cm=False,
    save_dir="confusion_matrices",
    run_id=None,
    architecture=None,
):
    predicted_labels = best_model.predict(X_test)
    if MTL == True:
        predicted_labels = predicted_labels[0]
    predicted_labels[predicted_labels > 0.5] = 1
    predicted_labels[predicted_labels <= 0.5] = 0
    actual_labels = y_test
    if show_cm:
        os.makedirs(save_dir, exist_ok=True)
        cm = confusion_matrix(actual_labels, predicted_labels)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Fresh", "Rotten"])
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix - Task 1 (Freshness)")
        plt.tight_layout()
        architecture_str = f"_{architecture}" if architecture else ""
        filename = (
            f"confusion_matrix_t2_{architecture_str}_run_{run_id}.png"
            if run_id
            else f"confusion_matrix_t2_{architecture_str}.png"
        )
        plt.savefig(os.path.join(save_dir, filename))
        plt.show()

    precision = precision_score(actual_labels, predicted_labels)
    recall = recall_score(actual_labels, predicted_labels)
    f1 = f1_score(actual_labels, predicted_labels)
    accuracy = accuracy_score(actual_labels, predicted_labels)
    return precision, recall, f1, accuracy


def evaluate_t2(
    X_test,
    y_test,
    best_model,
    MTL=False,
    show_cm=False,
    save_dir="confusion_matrices",
    run_id=None,
    architecture=None,
):
    predicted_labels_vec = best_model.predict(X_test)
    if MTL == True:
        predicted_labels = predicted_labels_vec[1]
    actual_labels = []
    predicted_labels = []
    for i in range(y_test.shape[0]):
        actual_labels.append(np.argmax(y_test[i]))
        if MTL == True:
            predicted_labels.append(np.argmax(predicted_labels_vec[1][i]))
        else:
            predicted_labels.append(np.argmax(predicted_labels_vec[i]))

    if show_cm:
        os.makedirs(save_dir, exist_ok=True)
        cm = confusion_matrix(actual_labels, predicted_labels)
        disp = ConfusionMatrixDisplay(
            cm,
            display_labels=[
                "Apple",
                "Banana",
                "Grape",
                "Guava",
                "Jujube",
                "Orange",
                "Pomegranate",
                "Strawberry",
            ],
        )
        disp.plot(xticks_rotation=45, cmap="Blues")
        plt.title("Confusion Matrix - Task 2 (Fruit Category)")
        plt.tight_layout()
        architecture_str = f"_{architecture}" if architecture else ""
        filename = (
            f"confusion_matrix_t1_{architecture_str}_run_{run_id}.png"
            if run_id
            else f"confusion_matrix_t1_{architecture_str}.png"
        )
        plt.savefig(os.path.join(save_dir, filename))
        plt.show()

    precision = precision_score(actual_labels, predicted_labels, average="macro")
    recall = recall_score(actual_labels, predicted_labels, average="macro")
    f1 = f1_score(actual_labels, predicted_labels, average="macro")
    accuracy = accuracy_score(actual_labels, predicted_labels)  # ,average='micro')
    return precision, recall, f1, accuracy


# define some common callbacks
lr_rate = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=6,
    verbose=1,
    mode="min",
    min_lr=0.00002,
    cooldown=2,
)

es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=4)


architectures = {
    "base": base_net,
    "cbam": base_net_cbam,
}

arch_metrics = {
    "cbam": {
        "T1_acc": [],
        "T1_F1": [],
        "T1_pre": [],
        "T1_rec": [],
        "T2_acc": [],
        "T2_F1": [],
        "T2_pre": [],
        "T2_rec": [],
        "time": [],
    },
    "base": {
        "T1_acc": [],
        "T1_F1": [],
        "T1_pre": [],
        "T1_rec": [],
        "T2_acc": [],
        "T2_F1": [],
        "T2_pre": [],
        "T2_rec": [],
        "time": [],
    },
}

# Iterate over both architectures in a single run and ensure consistent data split across both
for i in tqdm(range(RUNS)):
    # Make sure the data split is the same for each architecture run
    display(HTML(f"<h1>RUN #{i + 1}</h1>"))

    # Split the data once for this run
    (
        X_train,
        X_test,
        y_test_t1,
        y_test_t2,
        y_train_mtl,
        y_test_mtl,
        X_val,
        y_val_mtl,
    ) = split_data(0.3)

    # Store metrics for both architectures
    print(X_train.shape)
    # Run both architectures
    for arch_name, arch_fn in architectures.items():
        display(HTML(f"<h2>Training architecture: {arch_name.upper()}</h2>"))
        start_time = time.time()
        # Prepare model
        inputs = Input(shape=(100, 100, 3))
        x = arch_fn()(inputs)

        # Define the task-specific outputs
        fresh_output = Dense(EMBEDDING_DIM, activation="relu")(x)
        fresh_predict = Dense(1, activation="sigmoid", name="fresh")(fresh_output)

        cat_output = Dense(EMBEDDING_DIM, activation="relu")(x)
        cat_predict = Dense(8, activation="softmax", name="cat")(cat_output)

        model = Model(inputs=inputs, outputs=[fresh_predict, cat_predict])

        model.compile(
            loss={
                "fresh": "binary_crossentropy",
                "cat": "kl_divergence",
            },
            loss_weights={"fresh": 0.4, "cat": 0.6},
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            metrics=["accuracy", "accuracy"],
        )

        model_path = f"{arch_name}_mtl_run{i + 1}.keras"
        check_point = keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode="min",
        )

        # Train the model
        history = model.fit(
            X_train,
            y_train_mtl,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val_mtl),
            epochs=EPOCHS,
            callbacks=[check_point, lr_rate],
        )

        # Draw training curves
        draw_training_curve_MTL(history, architecture=arch_name)

        # Load the best model
        best_model = load_model(model_path)
        if arch_name == "base":
            layer_name = "separable_conv2d_1"
        else:
            layer_name = "separable_conv2d_3"
        print(best_model.summary(expand_nested=True))
        show_feature_maps(best_model, X_test, layer_name, n_images=5)

        # Evaluate Task 1 (Freshness)
        precision, recall, f1, accuracy = evaluate_t1(
            X_test,
            y_test_mtl[0],
            best_model=best_model,
            MTL=True,
            show_cm=True,
            run_id=i + 1,
            architecture=arch_name,
        )
        arch_metrics[arch_name]["T1_acc"].append(accuracy)
        arch_metrics[arch_name]["T1_F1"].append(f1)
        arch_metrics[arch_name]["T1_pre"].append(precision)
        arch_metrics[arch_name]["T1_rec"].append(recall)

        # Evaluate Task 2 (Fruit Category)
        precision, recall, f1, accuracy = evaluate_t2(
            X_test,
            y_test_mtl[1],
            best_model=best_model,
            MTL=True,
            show_cm=True,
            run_id=i + 1,
            architecture=arch_name,
        )
        arch_metrics[arch_name]["T2_acc"].append(accuracy)
        arch_metrics[arch_name]["T2_F1"].append(f1)
        arch_metrics[arch_name]["T2_pre"].append(precision)
        arch_metrics[arch_name]["T2_rec"].append(recall)

        # Track run time
        elapsed_time = time.time() - start_time
        arch_metrics[arch_name]["time"].append(elapsed_time)

        print(
            f"⏱️ Time taken for {arch_name.upper()} in RUN #{i + 1}: {elapsed_time:.2f} seconds"
        )

# Save metrics for both architectures in a single file
metrics_df = pd.DataFrame()
for arch_name in architectures.keys():
    for metric_name, values in arch_metrics[arch_name].items():
        metrics_df[f"{arch_name}_{metric_name}"] = values

# Save to Excel file after each run
metrics_df.to_excel(f"../../data/metrics_run_{i + 1}.xlsx", index=False)
