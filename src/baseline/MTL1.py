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
        "Mango",
        "Orange",
    ]
    X, Y = [], []
    z = []
    for cata in tqdm(os.listdir(path)):
        path_main = os.path.join(path, cata)

        for i, name in enumerate(quality):
            if quality[i] in cata:
                fresh_index = i
                break

        for i, name in enumerate(cat):
            if cat[i] in cata:
                cat_index = i
                print(f"Category Index: {cat_index}")
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
        label2.append(to_categorical(label[1], num_classes=5))

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


def split_data(test_size=0.3, seed=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, label_all, random_state=seed, test_size=test_size, stratify=label_all
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.3,
        random_state=seed,
        stratify=y_train,
    )

    # --- Augmentation ---
    datagen = ImageDataGenerator(
        brightness_range=(0.7, 1.3),
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    datagen.fit(X_train)

    # Initialize lists for augmented data
    augmented_images = []
    augmented_labels_t1 = []
    augmented_labels_t2 = []
    print(f"X_train:{len(X_train)}")
    print(f"y_train:{len(y_train)}")
    # Perform augmentation
    for _ in range(6):  # Generate 6 augmented copies
        aug_iter = datagen.flow(
            X_train,
            y_train,
            batch_size=X_train.shape[0],
            shuffle=False,
        )
        aug_images, aug_labels = next(aug_iter)
        augmented_images.append(aug_images)

        # Process labels for the two tasks (freshness and category)
        for label in aug_labels:
            a = [0] * 5
            augmented_labels_t1.append(int(label[0]))  # Freshness task (binary)
            a[int(label[1])] = 1  # Fruit category task (one-hot encoding)
            augmented_labels_t2.append(a)

    X_train_aug = np.concatenate(augmented_images, axis=0)
    augmented_labels_t1 = np.array(augmented_labels_t1)
    augmented_labels_t2 = np.array(augmented_labels_t2)

    y_test_t1 = []
    y_test_t2 = []
    for i in y_test:
        a = [0] * 5
        y_test_t1.append(int(i[0]))
        a[int(i[1])] = 1
        y_test_t2.append(a)
    y_test_t1 = np.array(y_test_t1)
    y_test_t2 = np.array(y_test_t2)

    y_val_t1 = []
    y_val_t2 = []
    for i in y_val:
        y_val_t1.append(int(i[0]))
        a = [0] * 5
        a[int(i[1])] = 1
        y_val_t2.append(a)
    y_val_t1 = np.array(y_val_t1)
    y_val_t2 = np.array(y_val_t2)

    y_test_mtl = [y_test_t1, y_test_t2]
    y_val_mtl = [y_val_t1, y_val_t2]
    y_train_aug_mtl = [augmented_labels_t1, augmented_labels_t2]
    print(f"X_train:{len(X_train_aug)}")
    print(f"y_train:{len(y_train_aug_mtl)}")

    return (
        X_train_aug,
        X_test,
        y_test_t1,
        y_test_t2,
        y_train_aug_mtl,
        y_test_mtl,
        X_val,
        y_val_mtl,
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
        input_shape=(100, 100, 3),
        weights=None,
        pooling="avg",
    )
    inputs = Input(shape=(100, 100, 3))
    x = base_model(inputs, training=False)
    x = Dropout(0.5)(x)  # Regularization
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
    start_time = time.time()

    # Split the data once for this run
    (
        X_train_aug,
        X_test,
        y_test_t1,
        y_test_t2,
        y_train_aug_mtl,
        y_test_mtl,
        X_val,
        y_val_mtl,
    ) = split_data(0.3)

    # Store metrics for both architectures
    print(X_train_aug.shape)
    # Run both architectures
    for arch_name, arch_fn in architectures.items():
        display(HTML(f"<h2>Training architecture: {arch_name.upper()}</h2>"))

        # Prepare model
        inputs = Input(shape=(100, 100, 3))
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

        # Train the model
        history = model.fit(
            X_train_aug,
            y_train_aug_mtl,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val_mtl),
            epochs=EPOCHS,
            callbacks=[check_point],
        )

        # Draw training curves
        draw_training_curve_MTL(history)

        # Load the best model
        best_model = load_model(model_path)

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
