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
import os
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
)
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

os.environ["KERAS_BACKEND"] = "torch"  # switch to torch backend


torch.cuda.is_available()

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
            img = cv2.resize(img, (300, 300))
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
        "Mango",
        "Orange",
    ]
    X_100 = []  # For 100x100 images
    X_300 = []  # For 300x300 images
    Y = []
    z1 = []
    z2 = []

    for cata in tqdm(os.listdir(path)):
        path_main = os.path.join(path, cata)

        for i, name in enumerate(quality):
            if quality[i] in cata:
                fresh_index = i
                break

        for i, name in enumerate(cat):
            if cat[i] in cata:
                cat_index = i
                break

        for img_name in os.listdir(path_main):
            # Load and process image
            img = cv2.imread(os.path.join(path_main, img_name))
            img_100 = cv2.resize(img, (100, 100))  # Resize to 100x100
            img_300 = cv2.resize(img, (300, 300))  # Resize to 300x300

            img_100 = cv2.cvtColor(img_100, cv2.COLOR_BGR2RGB)
            img_300 = cv2.cvtColor(img_300, cv2.COLOR_BGR2RGB)

            # Append both resized images to their respective lists

            z1.append([img_100, [fresh_index, cat_index]])
            z2.append([img_300, [fresh_index, cat_index]])
    print("Loading and shuffling image data ... ...")
    shuffle(z1)
    shuffle(z2)
    for images, labels in z1:
        X_100.append(images)
        Y.append(labels)

    for images, labels in z2:
        X_300.append(images)

    # Convert to numpy arrays and normalize
    X_100 = np.array(X_100) / 255.0
    X_300 = np.array(X_300) / 255.0

    label1 = []
    label2 = []

    for label in Y:
        label1.append(label[0])  # Freshness
        label2.append(to_categorical(label[1], num_classes=5))  # One-hot category

    Y = [np.array(label1), np.array(label2)]

    return X_100, X_300, Y


if not os.path.exists(PICKLE_FILE):
    X_100, X_300, Y = load_mtl_data(path=DATA_PATH)

    # Create label_all for both 100x100 and 300x300 images
    label_all_100 = []
    for i in range(len(X_100)):
        label_all_100.append(str(Y[0][i]) + str(np.argmax(Y[1][i])))

    label_all_100 = np.array(label_all_100)

    label_all_300 = []
    for i in range(len(X_300)):
        label_all_300.append(str(Y[0][i]) + str(np.argmax(Y[1][i])))

    label_all_300 = np.array(label_all_300)

    # Save both image sets and labels in pickle file
    pickle.dump(
        [X_100, X_300, Y, label_all_100, label_all_300], open(PICKLE_FILE, "wb")
    )

else:
    # Load the pickle file containing both image sizes and labels
    X_100, X_300, Y, label_all_100, label_all_300 = pickle.load(open(PICKLE_FILE, "rb"))
    print("X_100 shape from pickle:", X_100.shape)
    print("X_300 shape from pickle:", X_300.shape)


def split_data(test_size=0.3, seed=None):
    # Split data for 100x100 images
    X_train_100, X_test_100, y_train_100, y_test_100 = train_test_split(
        X_100,
        label_all_100,
        random_state=seed,
        test_size=test_size,
        stratify=label_all_100,
    )
    X_train_100, X_val_100, y_train_100, y_val_100 = train_test_split(
        X_train_100,
        y_train_100,
        test_size=0.3,
        random_state=seed,
        stratify=y_train_100,
    )

    # Split data for 300x300 images
    X_train_300, X_test_300, y_train_300, y_test_300 = train_test_split(
        X_300,
        label_all_300,
        random_state=seed,
        test_size=test_size,
        stratify=label_all_300,
    )
    X_train_300, X_val_300, y_train_300, y_val_300 = train_test_split(
        X_train_300,
        y_train_300,
        test_size=0.3,
        random_state=seed,
        stratify=y_train_300,
    )

    # --- Augmentation for 100x100 images ---
    datagen_100 = ImageDataGenerator(
        brightness_range=(0.7, 1.3),
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    datagen_100.fit(X_train_100)

    augmented_images_100 = []
    augmented_labels_100_t1 = []
    augmented_labels_100_t2 = []
    for _ in range(6):  # Generate 6 augmented copies for 100x100 images
        aug_iter = datagen_100.flow(
            X_train_100,
            y_train_100,
            batch_size=X_train_100.shape[0],
            shuffle=False,
        )
        aug_images, aug_labels = next(aug_iter)
        augmented_images_100.append(aug_images)
        for label in aug_labels:
            augmented_labels_100_t1.append(int(label[0]))
            a = [0] * 5
            a[int(label[1])] = 1
            augmented_labels_100_t2.append(a)

    X_train_100_aug = np.concatenate(augmented_images_100, axis=0)
    augmented_labels_100_t1 = np.array(augmented_labels_100_t1)
    augmented_labels_100_t2 = np.array(augmented_labels_100_t2)

    # --- Augmentation for 300x300 images ---
    datagen_300 = ImageDataGenerator(
        brightness_range=(0.7, 1.3),
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    datagen_300.fit(X_train_300)

    augmented_images_300 = []
    augmented_labels_300_t1 = []
    augmented_labels_300_t2 = []
    for _ in range(6):  # Generate 6 augmented copies for 300x300 images
        aug_iter = datagen_300.flow(
            X_train_300,
            y_train_300,
            batch_size=X_train_300.shape[0],
            shuffle=False,
        )
        aug_images, aug_labels = next(aug_iter)
        augmented_images_300.append(aug_images)
        for label in aug_labels:
            augmented_labels_300_t1.append(int(label[0]))
            a = [0] * 5
            a[int(label[1])] = 1
            augmented_labels_300_t2.append(a)

    X_train_300_aug = np.concatenate(augmented_images_300, axis=0)
    augmented_labels_300_t1 = np.array(augmented_labels_300_t1)
    augmented_labels_300_t2 = np.array(augmented_labels_300_t2)

    # --- Process the validation and test sets ---
    y_test_100_t1 = []
    y_test_100_t2 = []
    for i in y_test_100:
        y_test_100_t1.append(int(i[0]))
        a = [0] * 5
        a[int(i[1])] = 1
        y_test_100_t2.append(a)
    y_test_100_t1 = np.array(y_test_100_t1)
    y_test_100_t2 = np.array(y_test_100_t2)

    y_test_300_t1 = []
    y_test_300_t2 = []
    for i in y_test_300:
        y_test_300_t1.append(int(i[0]))
        a = [0] * 5
        a[int(i[1])] = 1
        y_test_300_t2.append(a)
    y_test_300_t1 = np.array(y_test_300_t1)
    y_test_300_t2 = np.array(y_test_300_t2)

    # Validation sets
    y_val_100_t1 = []
    y_val_100_t2 = []
    for i in y_val_100:
        y_val_100_t1.append(int(i[0]))
        a = [0] * 5
        a[int(i[1])] = 1
        y_val_100_t2.append(a)
    y_val_100_t1 = np.array(y_val_100_t1)
    y_val_100_t2 = np.array(y_val_100_t2)

    y_val_300_t1 = []
    y_val_300_t2 = []
    for i in y_val_300:
        y_val_300_t1.append(int(i[0]))
        a = [0] * 5
        a[int(i[1])] = 1
        y_val_300_t2.append(a)
    y_val_300_t1 = np.array(y_val_300_t1)
    y_val_300_t2 = np.array(y_val_300_t2)

    # Return training, validation, and test data for both image sizes
    y_train_100_mtl = [augmented_labels_100_t1, augmented_labels_100_t2]
    y_train_300_mtl = [augmented_labels_300_t1, augmented_labels_300_t2]
    y_test_100_mtl = [y_test_100_t1, y_test_100_t2]
    y_test_300_mtl = [y_test_300_t1, y_test_300_t2]
    y_val_100_mtl = [y_val_100_t1, y_val_100_t2]
    y_val_300_mtl = [y_val_300_t1, y_val_300_t2]

    return (
        X_train_100_aug,
        X_train_300_aug,
        X_test_100,
        X_test_300,
        X_val_100,
        X_val_300,
        y_test_100_mtl,
        y_test_300_mtl,
        y_train_100_mtl,
        y_train_300_mtl,
        y_val_100_mtl,
        y_val_300_mtl,
    )


# global settings

RUNS = 1  # the full experiment runs 50 times.

K_NUM = 32  # kernel num, e.g., 64
EMBEDDING_DIM = 64  # the final embedding size, e.g., 128

EPOCHS = 15  # max epochs to train for various models
BATCH_SIZE = 32  # 18 default


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


def efficientnetb3_net():
    base_model = EfficientNetB3(
        include_top=False,
        input_shape=(300, 300, 3),
        weights=None,
        pooling="avg",
    )
    inputs = Input(shape=(300, 300, 3))
    x = base_model(inputs, training=False)
    x = Dropout(0.5)(x)
    return Model(inputs, x)


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


def draw_training_curve_MTL(history):
    plt.figure(1, figsize=(20, 8))
    # plt.title('MTL')
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
                "Mango",
                "Orange",
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
    mode="max",
    min_lr=0.00002,
    cooldown=2,
)

es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=4)


architectures = {
    "efficientnet": efficientnetb3_net,
    "base": base_net,
}

arch_metrics = {
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
    "efficientnet": {
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
    start_time = time.time()

    # Split the data once for this run (this should return split data for both image sizes)
    (
        X_train_aug_100,
        X_train_aug_300,
        X_test_100,
        X_test_300,
        X_val_100,
        X_val_300,
        y_test_100_mtl,
        y_test_300_mtl,
        y_train_100_mtl,
        y_train_300_mtl,
        y_val_100_mtl,
        y_val_300_mtl,
    ) = split_data(0.3)

    # Store metrics for both architectures
    print("Training images shape for 100x100:", X_train_aug_100.shape)
    print("Training images shape for 300x300:", X_train_aug_300.shape)

    # Run both architectures
    for arch_name, arch_fn in architectures.items():
        display(HTML(f"<h2>Training architecture: {arch_name.upper()}</h2>"))

        # Choose the correct input shape for the architecture
        input_shape = (100, 100, 3) if arch_name == "base" else (300, 300, 3)
        print(arch_name)
        print(input_shape)
        # Prepare the model with the correct input shape
        inputs = Input(shape=input_shape)
        x = arch_fn()(inputs)

        # Define the task-specific outputs
        fresh_output = Dense(EMBEDDING_DIM, activation="relu")(x)
        fresh_predict = Dense(1, activation="sigmoid", name="fresh")(fresh_output)

        cat_output = Dense(EMBEDDING_DIM, activation="relu")(x)
        cat_predict = Dense(5, activation="softmax", name="cat")(cat_output)

        model = Model(inputs=inputs, outputs=[fresh_predict, cat_predict])

        model.compile(
            loss={
                "fresh": "binary_crossentropy",
                "cat": "kl_divergence",
            },
            loss_weights={"fresh": 0.4, "cat": 0.6},
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
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

        # Train the model (choose the image size for training)
        X_train_aug = (
            X_train_aug_100 if input_shape == (100, 100, 3) else X_train_aug_300
        )
        y_train_mtl = (
            y_train_100_mtl if input_shape == (100, 100, 3) else y_train_300_mtl
        )
        X_val = X_val_100 if input_shape == (100, 100, 3) else X_val_300
        y_val_mtl = y_val_100_mtl if input_shape == (100, 100, 3) else y_val_300_mtl

        history = model.fit(
            X_train_aug,
            y_train_mtl,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val_mtl),
            epochs=EPOCHS,
            callbacks=[check_point],
        )

        # Draw training curves
        draw_training_curve_MTL(history)

        # Load the best model
        best_model = load_model(model_path)

        # Evaluate Task 1 (Freshness) for the appropriate image size
        precision, recall, f1, accuracy = evaluate_t1(
            X_test_100 if input_shape == (100, 100, 3) else X_test_300,
            y_test_100_mtl[0] if input_shape == (100, 100, 3) else y_test_300_mtl[0],
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

        # Evaluate Task 2 (Fruit Category) for the appropriate image size
        precision, recall, f1, accuracy = evaluate_t2(
            X_test_100 if input_shape == (100, 100, 3) else X_test_300,
            y_test_100_mtl[1] if input_shape == (100, 100, 3) else y_test_300_mtl[1],
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
