import os
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import train_test_split
from PIL import Image
import albumentations as A
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt

# Load CSV File
df = pd.read_csv("../../data/Dataset/Efficientnetb3/fruit.csv")

# Create a mapping from fruit names to integer labels
fruit_label_mapping = {"banana": 0, "apple": 1, "orange": 2}

# Define Albumentations augmentations
albumentations_transform = A.Compose(
    [
        A.Rotate(limit=30, border_mode=cv2.BORDER_REPLICATE, p=1),
        A.CoarseDropout(
            num_holes_range=(1, 10),
            hole_height_range=(20, 50),
            hole_width_range=(20, 50),
            p=1,
        ),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
    ]
)

# Define standard PyTorch transforms
torch_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Define Custom Dataset Class
class FruitFreshnessDataset(Dataset):
    def __init__(
        self,
        image_paths,
        fruit_labels,
        freshness_labels,
        transform=None,
        apply_augmentations=False,
        num_augments=6,
    ):
        self.image_paths = image_paths
        self.fruit_labels = [fruit_label_mapping[label] for label in fruit_labels]
        self.freshness_labels = freshness_labels
        self.transform = transform
        self.apply_augmentations = apply_augmentations
        self.num_augments = num_augments if apply_augmentations else 1
        # Define augmentation transformations
        self.augmentations = [
            A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, p=1),
            A.CoarseDropout(
                num_holes_range=(1, 10),
                hole_height_range=(50, 50),
                hole_width_range=(50, 50),
                p=1,
            ),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=1
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=1),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
        ]

    def __len__(self):
        return len(self.image_paths) * self.num_augments

    def __getitem__(self, idx):
        original_idx = idx // self.num_augments
        augment_idx = idx % self.num_augments
        image_path = self.image_paths[original_idx]
        fruit_label = self.fruit_labels[original_idx]
        freshness_label = self.freshness_labels[original_idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.apply_augmentations:
            augmented = self.augmentations[augment_idx](image=image)
            image = augmented["image"]

        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)

        return (
            image,
            torch.tensor(fruit_label, dtype=torch.long),
            torch.tensor(freshness_label, dtype=torch.float32),
        )


# Prepare Dataset
image_directory = "../../data/Dataset/Efficientnetb3/Images"
image_paths = [
    os.path.join(image_directory, name) for name in df["image_name"].tolist()
]
fruit_labels = df["fruit_type"].tolist()
freshness_labels = df["grading"].tolist()

# Split Data
train_img, temp_img, train_fruit, temp_fruit, train_fresh, temp_fresh = (
    train_test_split(
        image_paths,
        fruit_labels,
        freshness_labels,
        test_size=0.3,
        random_state=42,
        stratify=fruit_labels,
    )
)
val_img, test_img, val_fruit, test_fruit, val_fresh, test_fresh = train_test_split(
    temp_img,
    temp_fruit,
    temp_fresh,
    test_size=0.5,
    random_state=42,
    stratify=temp_fruit,
)

train_dataset = FruitFreshnessDataset(
    train_img,
    train_fruit,
    train_fresh,
    transform=torch_transform,
    apply_augmentations=True,
    num_augments=6,
)
val_dataset = FruitFreshnessDataset(
    val_img, val_fruit, val_fresh, transform=torch_transform, apply_augmentations=False
)
test_dataset = FruitFreshnessDataset(
    test_img,
    test_fruit,
    test_fresh,
    transform=torch_transform,
    apply_augmentations=False,
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

print(len(train_loader.dataset))
print(len(val_loader.dataset))
print(len(test_loader.dataset))


# Define Multi-Task Model
class MultiTaskEfficientNet(nn.Module):
    def __init__(self, num_classes=3):
        super(MultiTaskEfficientNet, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained("efficientnet-b3")
        in_features = self.efficientnet._fc.in_features
        self.efficientnet._fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        self.regressor = nn.Sequential(
            nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1)
        )

    def forward(self, x):
        shared_features = self.efficientnet(x)
        fruit_output = self.classifier(shared_features)
        freshness_output = self.regressor(shared_features)
        return fruit_output, freshness_output


# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTaskEfficientNet().to(device)

# Define Losses
classification_loss_fn = nn.CrossEntropyLoss()
regression_loss_fn = nn.MSELoss()

# Define Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-5)

# Training Loop
num_epochs = 8
best_val_loss = float("inf")

train_class_losses, train_reg_losses = [], []
val_class_losses, val_reg_losses = [], []

for epoch in range(num_epochs):
    model.train()
    train_class_loss, train_reg_loss = 0.0, 0.0

    for images, fruit_labels, freshness_labels in train_loader:
        images, fruit_labels, freshness_labels = (
            images.to(device),
            fruit_labels.to(device),
            freshness_labels.to(device),
        )
        optimizer.zero_grad()

        # Forward pass
        fruit_preds, freshness_preds = model(images)

        # Compute losses
        loss_class = classification_loss_fn(fruit_preds, fruit_labels)
        loss_reg = regression_loss_fn(freshness_preds.squeeze(), freshness_labels)

        alpha = 0.3
        beta = 0.7
        loss = alpha * loss_class + beta * loss_reg
        loss.backward()
        optimizer.step()

        train_class_loss += loss_class.item()
        train_reg_loss += loss_reg.item()

    # Validation phase
    model.eval()
    val_class_loss, val_reg_loss = 0.0, 0.0

    with torch.no_grad():
        for images, fruit_labels, freshness_labels in val_loader:
            images, fruit_labels, freshness_labels = (
                images.to(device),
                fruit_labels.to(device),
                freshness_labels.to(device),
            )
            fruit_preds, freshness_preds = model(images)
            loss_class = classification_loss_fn(fruit_preds, fruit_labels)
            loss_reg = regression_loss_fn(freshness_preds.squeeze(), freshness_labels)
            val_class_loss += loss_class.item()
            val_reg_loss += loss_reg.item()

    # Average losses
    avg_train_class_loss = train_class_loss / len(train_loader)
    avg_train_reg_loss = train_reg_loss / len(train_loader)
    avg_val_class_loss = val_class_loss / len(val_loader)
    avg_val_reg_loss = val_reg_loss / len(val_loader)

    train_class_losses.append(avg_train_class_loss)
    train_reg_losses.append(avg_train_reg_loss)
    val_class_losses.append(avg_val_class_loss)
    val_reg_losses.append(avg_val_reg_loss)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] - "
        f"Train Class Loss: {avg_train_class_loss:.4f}, Train Reg Loss: {avg_train_reg_loss:.4f}, "
        f"Val Class Loss: {avg_val_class_loss:.4f}, Val Reg Loss: {avg_val_reg_loss:.4f}"
    )


# Evaluate on Test Data
model.eval()
test_class_loss, test_reg_loss = 0.0, 0.0
all_fruit_preds, all_fruit_labels = [], []


with torch.no_grad():
    for images, fruit_labels, freshness_labels in test_loader:
        images, fruit_labels, freshness_labels = (
            images.to(device),
            fruit_labels.to(device),
            freshness_labels.to(device),
        )
        fruit_preds, freshness_preds = model(images)
        loss_class = classification_loss_fn(fruit_preds, fruit_labels)
        loss_reg = regression_loss_fn(freshness_preds.squeeze(), freshness_labels)
        test_class_loss += loss_class.item()
        test_reg_loss += loss_reg.item()

        # Store predictions and labels
        all_fruit_preds.extend(torch.argmax(fruit_preds, dim=1).cpu().numpy())
        all_fruit_labels.extend(fruit_labels.cpu().numpy())


accuracy = accuracy_score(all_fruit_labels, all_fruit_preds)
precision = precision_score(all_fruit_labels, all_fruit_preds, average="macro")
recall = recall_score(all_fruit_labels, all_fruit_preds, average="macro")

print(
    f"Test Class Loss: {test_class_loss / len(test_loader):.4f}, Test Reg Loss: {test_reg_loss / len(test_loader):.4f}"
)

print(
    f"Classification Accuracy: {accuracy:.4f}, "
    f"Precision: {precision:.4f}, "
    f"Recall: {recall:.4f}"
)

fig, ax = plt.subplots(figsize=(6, 2))
ax.axis("tight")
ax.axis("off")

# Define table data
table_data = [
    ["Metric", "Value"],
    ["Accuracy", f"{accuracy:.4f}"],
    ["Precision", f"{precision:.4f}"],
    ["Recall", f"{recall:.4f}"],
]
# Create the table
table = ax.table(cellText=table_data, cellLoc="center", loc="center")
# Save as an image
plt.savefig("EfficientNetB3_Test_Results_Table.png")
plt.show()

# Plot Classification Loss Curve
plt.figure(figsize=(8, 6))
plt.plot(
    range(1, num_epochs + 1),
    train_class_losses,
    label="Train Classification Loss",
    marker="o",
)
plt.plot(
    range(1, num_epochs + 1),
    val_class_losses,
    label="Validation Classification Loss",
    marker="o",
)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training & Validation Classification Loss Curve")
plt.legend()
plt.grid()
plt.savefig("EfficientNetB3_Classification_Loss.png")
plt.show()

# Plot Regression Loss Curve
plt.figure(figsize=(8, 6))
plt.plot(
    range(1, num_epochs + 1),
    train_reg_losses,
    label="Train Regression Loss",
    marker="o",
)
plt.plot(
    range(1, num_epochs + 1),
    val_reg_losses,
    label="Validation Regression Loss",
    marker="o",
)
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title("Training & Validation Regression Loss Curve")
plt.legend()
plt.grid()
plt.savefig("EfficientNetB3_Regression_Loss.png")
plt.show()


# Save Model
torch.save(model.state_dict(), "EfficientNetB3_Fruit_Grading.pth")
