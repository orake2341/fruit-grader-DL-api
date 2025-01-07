import torch
from torchvision import models
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# Define the model structure
vgg11 = models.vgg11(pretrained=False)  # Don't load pretrained weights again
vgg11.classifier = nn.Sequential(
    nn.Linear(25088, 4069),
    nn.ReLU(),
    nn.Dropout(0.6),
    nn.Linear(4069, 4069),
    nn.ReLU(),
    nn.Dropout(0.6),
    nn.Linear(4069, 1096),
    nn.ReLU(),
    nn.Dropout(0.6),
    nn.Linear(1096, 296),
    nn.ReLU(),
    nn.Linear(296, 56),
    nn.ReLU(),
    nn.Linear(56, 1),
)

# Load the saved model weights
vgg11.load_state_dict(torch.load("../models/vgg11_freshness_model.pth"))
vgg11.eval()  # Set the model to evaluation mode

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg11 = vgg11.to(device)

# Define preprocessing transformations (same as during training)
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Load and preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Ensure it's RGB
    return transform(image).unsqueeze(0)  # Add batch dimension


# Example: Preprocess a test image
image_path = "../data/Augmented Dataset/Banana100_aug1.jpg"
input_tensor = preprocess_image(image_path).to(device)

# Perform prediction
with torch.no_grad():  # Disable gradient calculation for inference
    output = vgg11(input_tensor).squeeze().item()  # Get the single output value

# Output the predicted grade
print(f"Predicted Freshness Grade: {output:.2f}")
