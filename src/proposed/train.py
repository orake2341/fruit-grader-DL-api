import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations (no augmentation)
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load dataset
train_dir = "../../data/VGG16/train"
valid_dir = "../../data/VGG16/valid"
test_dir = "../../data/VGG16/test"

train_dataset = ImageFolder(train_dir, transform=transform)
valid_dataset = ImageFolder(valid_dir, transform=transform)
test_dataset = ImageFolder(test_dir, transform=transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


# Define model
class VGG16Classifier(nn.Module):
    def __init__(self, num_classes):
        super(VGG16Classifier, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg16.classifier[6] = nn.Linear(
            4096, num_classes
        )  # Adjust for class count

    def forward(self, x):
        return self.vgg16(x)


num_classes = len(train_dataset.classes)
model = VGG16Classifier(num_classes).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

# Training loop
num_epochs = 8
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100.0 * correct / total

    # Validation phase
    model.eval()
    valid_loss, correct_valid, total_valid = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            _, predicted = outputs.max(1)
            total_valid += labels.size(0)
            correct_valid += predicted.eq(labels).sum().item()

    valid_loss /= len(valid_loader)
    valid_acc = 100.0 * correct_valid / total_valid

    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%"
    )

# Testing phase
model.eval()
test_loss, correct_test, total_test = 0.0, 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total_test += labels.size(0)
        correct_test += predicted.eq(labels).sum().item()

test_loss /= len(test_loader)
test_acc = 100.0 * correct_test / total_test
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

# Save model
torch.save(model.state_dict(), "VGG16_Fruit_Classifier.pth")
print("Training complete and model saved.")
