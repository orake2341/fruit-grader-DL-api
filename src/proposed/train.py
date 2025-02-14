import os
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import train_test_split

# Load CSV File
df = pd.read_csv("../../data/fresh.csv")


# Create a mapping from fruit names to integer labels
fruit_label_mapping = {
    "Apple": 0,
    "Banana": 1,
    "Orange": 2,
}  # Adjust based on your dataset


class FruitFreshnessDataset(Dataset):
    def __init__(self, image_paths, fruit_labels, freshness_labels, transform=None):
        self.image_paths = image_paths
        self.fruit_labels = [
            fruit_label_mapping[label] for label in fruit_labels
        ]  # Convert to integers
        self.freshness_labels = freshness_labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        fruit_label = self.fruit_labels[idx]
        freshness_label = self.freshness_labels[idx]

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return (
            image,
            torch.tensor(fruit_label, dtype=torch.long),  # Integer for classification
            torch.tensor(freshness_label, dtype=torch.float32),  # Float for regression
        )


# Define Transforms
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Prepare Dataset
image_directory = "../../data/Fresh"
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
    train_img, train_fruit, train_fresh, transform=transform
)
val_dataset = FruitFreshnessDataset(val_img, val_fruit, val_fresh, transform=transform)
test_dataset = FruitFreshnessDataset(
    test_img, test_fruit, test_fresh, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


# Define Multi-Task Model
class MultiTaskEfficientNet(nn.Module):
    def __init__(self, num_classes=3):
        super(MultiTaskEfficientNet, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained("efficientnet-b3")

        # Shared feature extractor
        in_features = self.efficientnet._fc.in_features
        self.efficientnet._fc = nn.Identity()

        # Classification head (Fruit Type)
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        # Regression head (Freshness Grading)
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
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

# Training Loop
num_epochs = 8
best_val_loss = float("inf")

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

        loss = loss_class + loss_reg
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

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] - "
        f"Train Class Loss: {avg_train_class_loss:.4f}, Train Reg Loss: {avg_train_reg_loss:.4f}, "
        f"Val Class Loss: {avg_val_class_loss:.4f}, Val Reg Loss: {avg_val_reg_loss:.4f}"
    )


# Evaluate on Test Data
model.eval()
test_class_loss, test_reg_loss = 0.0, 0.0

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

print(
    f"Test Class Loss: {test_class_loss / len(test_loader):.4f}, Test Reg Loss: {test_reg_loss / len(test_loader):.4f}"
)

# Save Model
torch.save(model.state_dict(), "EfficientNetB3_Fruit_Grading.pth")
