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

# Load the CSV file
df = pd.read_csv(
    r"D:\Repository\fruit-detection-grading\data\cropped\labels\vgg11_freshness_data.csv"
)


# Define a custom dataset class
class FreshnessDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)


# Preprocessing transformations
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Prepare dataset
image_directory = "../data/cropped/images"
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

train_dataset = FreshnessDataset(train_image_paths, train_labels, transform=transform)
val_dataset = FreshnessDataset(val_image_paths, val_labels, transform=transform)
test_dataset = FreshnessDataset(test_image_paths, test_labels, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

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
    nn.Dropout(0.5),
    nn.Linear(1096, 296),
    nn.ReLU(),
    nn.Linear(296, 56),
    nn.ReLU(),
    nn.Linear(56, 1),  # Single output for regression (freshness grade)
)

# Define the loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = torch.optim.Adam(vgg11.parameters(), lr=0.0001, weight_decay=1e-5)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg11 = vgg11.to(device)

# Training loop with validation
num_epochs = 8
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

    print(
        f"Epoch [{epoch+1}/{num_epochs}], "
        f"Train Loss: {train_loss/len(train_loader):.4f}, "
        f"Validation Loss: {val_loss/len(val_loader):.4f}"
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

print(f"Test Loss: {test_loss/len(test_loader):.4f}")

# Save the trained model
torch.save(vgg11.state_dict(), "vgg11_freshness_model.pth")
