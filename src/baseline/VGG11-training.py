import os
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
df = pd.read_csv("../../data/Dataset/Vgg11/banana.csv")
image_directory = "../../data/Dataset/Vgg11/banana"

# Correctly filter images from different videos
video1_data = df[df["image_name"].str.contains(r"\(a\)", regex=True)]
video2_data = df[df["image_name"].str.contains(r"\(b\)", regex=True)]
video3_data = df[df["image_name"].str.contains(r"\(c\)", regex=True)]

train_data = pd.concat([video1_data, video2_data])
video3_selected = video3_data.sample(n=175, random_state=42)
val_data, test_data = train_test_split(video3_selected, test_size=0.5, random_state=42)


# Define standard PyTorch transforms (for normalization & tensor conversion)
torch_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Define a custom dataset class
class FreshnessDataset(Dataset):
    def __init__(
        self,
        image_paths,
        labels,
        transform=None,
        apply_augmentations=False,
        num_augments=6,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.apply_augmentations = apply_augmentations
        self.num_augments = num_augments if apply_augmentations else 1

        # Define augmentation transformations
        self.augmentations = [
            A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, p=1),
            A.CoarseDropout(
                num_holes_range=(5, 10),
                hole_height_range=(50, 50),
                hole_width_range=(50, 50),
                p=1,
            ),
            A.HueSaturationValue(
                hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1
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
        label = self.labels[original_idx]

        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Apply only one augmentation per instance (if training)
        if self.apply_augmentations:
            augmented = self.augmentations[augment_idx](image=image)
            image = augmented["image"]

        # Convert image to PIL format (for PyTorch transforms)
        image = Image.fromarray(image)

        # Apply PyTorch transforms (resize, normalize, tensor conversion)
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)


# Prepare dataset

image_paths = [
    os.path.join(image_directory, name) for name in df["image_name"].tolist()
]


train_image_paths = [
    os.path.join(image_directory, name) for name in train_data["image_name"]
]
val_image_paths = [
    os.path.join(image_directory, name) for name in val_data["image_name"]
]
test_image_paths = [
    os.path.join(image_directory, name) for name in test_data["image_name"]
]

train_labels = train_data["grading"].tolist()
val_labels = val_data["grading"].tolist()
test_labels = test_data["grading"].tolist()

print(df.head())
# Create dataset instances
train_dataset = FreshnessDataset(
    train_image_paths,
    train_labels,
    transform=torch_transform,
    apply_augmentations=True,
    num_augments=6,
)
val_dataset = FreshnessDataset(
    val_image_paths, val_labels, transform=torch_transform, apply_augmentations=False
)
test_dataset = FreshnessDataset(
    test_image_paths, test_labels, transform=torch_transform, apply_augmentations=False
)

# Create DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

print(len(train_loader.dataset))
print(len(val_loader.dataset))
print(len(test_loader.dataset))


# # Define directories to save images
# save_dirs = {
#     "train": "saved_images/train",
#     "val": "saved_images/val",
#     "test": "saved_images/test",
# }

# # Create directories if they don't exist
# for dir_path in save_dirs.values():
#     os.makedirs(dir_path, exist_ok=True)


# # Function to save images from DataLoader
# def save_images(dataloader, save_dir):
#     index = 0  # Image index
#     for images, labels in dataloader:
#         for i in range(images.size(0)):  # Iterate through batch
#             img = images[i].permute(1, 2, 0).cpu().numpy()  # Convert to NumPy
#             img = img * np.array([0.229, 0.224, 0.225]) + np.array(
#                 [0.485, 0.456, 0.406]
#             )  # Unnormalize
#             img = np.clip(img, 0, 1)  # Clip to valid range

#             # Convert NumPy array back to PIL image
#             pil_img = Image.fromarray((img * 255).astype("uint8"))

#             # Save the image
#             img_path = os.path.join(save_dir, f"image_{index}.png")
#             pil_img.save(img_path)

#             index += 1  # Increment index


# # Save images from each DataLoader
# save_images(train_loader, save_dirs["train"])
# save_images(val_loader, save_dirs["val"])
# save_images(test_loader, save_dirs["test"])


# Load the VGG11 model
vgg11 = models.vgg11(pretrained=True)

# Freeze VGG11 feature extractor layers
for param in vgg11.features.parameters():
    param.requires_grad = False

# Modify the classifier
vgg11.classifier = nn.Sequential(
    nn.Linear(25088, 4069),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4069, 4069),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4069, 1096),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(1096, 296),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(296, 56),
    nn.ReLU(),
    nn.Linear(56, 1),
)

# Define the loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = torch.optim.Adam(vgg11.parameters(), lr=0.00001, weight_decay=5e-4)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg11 = vgg11.to(device)

# Training loop with validation
num_epochs = 15
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    # Training phase
    vgg11.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = vgg11(images).squeeze()
        loss = criterion(outputs, labels.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation phase
    vgg11.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = vgg11(images).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Train Loss: {avg_train_loss:.4f}, "
        f"Validation Loss: {avg_val_loss:.4f}"
    )

# Test phase
vgg11.eval()
test_loss = 0.0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = vgg11(images).squeeze()
        loss = criterion(outputs, labels)
        test_loss += loss.item()

avg_test_loss = test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss:.4f}")

# Save the trained model
torch.save(vgg11.state_dict(), "banana_vgg11_freshness_model.pth")

# Plot Training & Validation Loss
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss", marker="o")
plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss", marker="o")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title("Training & Validation Loss Curve")
plt.legend()
plt.grid()
plt.savefig("banana_training_loss.png")
plt.show()
