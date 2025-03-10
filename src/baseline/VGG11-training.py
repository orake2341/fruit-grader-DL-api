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

# Load the CSV file
df = pd.read_csv("../../data/Dataset/Banana.csv")

# Define Albumentations augmentations
albumentations_transform = A.Compose(
    [
        A.Rotate(limit=30, border_mode=cv2.BORDER_REPLICATE, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.HueSaturationValue(
            hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5
        ),
        A.CoarseDropout(
            num_holes_range=(1, 10),
            hole_height_range=(20, 50),
            hole_width_range=(20, 50),
            p=0.5,
        ),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    ]
)

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
    def __init__(self, image_paths, labels, transform=None, apply_augmentations=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.apply_augmentations = apply_augmentations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Apply augmentations only in training mode
        if self.apply_augmentations:
            augmented = albumentations_transform(image=image)
            image = augmented["image"]

        # Convert image to PIL format (for PyTorch transforms)
        image = Image.fromarray(image)

        # Apply PyTorch transforms (resize, normalize, tensor conversion)
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)


# Prepare dataset
image_directory = "../../data/Dataset/Banana"
image_paths = [
    os.path.join(image_directory, name) for name in df["image_name"].tolist()
]
labels = df["grading"].tolist()

# Split dataset into training, validation, and test
train_image_paths, temp_image_paths, train_labels, temp_labels = train_test_split(
    image_paths, labels, test_size=0.3, random_state=42
)
val_image_paths, test_image_paths, val_labels, test_labels = train_test_split(
    temp_image_paths, temp_labels, test_size=0.5, random_state=42
)

# Create dataset instances
train_dataset = FreshnessDataset(
    train_image_paths, train_labels, transform=torch_transform, apply_augmentations=True
)
val_dataset = FreshnessDataset(
    val_image_paths, val_labels, transform=torch_transform, apply_augmentations=False
)
test_dataset = FreshnessDataset(
    test_image_paths, test_labels, transform=torch_transform, apply_augmentations=False
)

# Create DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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
    nn.Linear(296, 56),
    nn.ReLU(),
    nn.Linear(56, 1),  # Single output for regression (freshness grade)
)

# Define the loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = torch.optim.Adam(vgg11.parameters(), lr=0.0001)


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=30, eta_min=1e-6
)
# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg11 = vgg11.to(device)

# Training loop with validation
num_epochs = 30
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    # Training phase
    vgg11.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = vgg11(images).squeeze()
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()

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
torch.save(vgg11.state_dict(), "Banana_vgg11_freshness_model.pth")

# Plot Training & Validation Loss
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss", marker="o")
plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss", marker="o")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title("Training & Validation Loss Curve")
plt.legend()
plt.grid()
plt.show()
