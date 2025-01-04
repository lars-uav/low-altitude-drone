import torch
import os
import numpy as np
from pathlib import Path
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import timm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import wandb

# Initialize a new run
wandb.login(key="afe8b8c0a3f1c1339a3daa9f619cb7c311218022")
wandb.init(project="paddy-disease-classification")

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths
train_path = Path('/kaggle/input/paddy-disease-classification/train_images')
test_path = Path('/kaggle/input/paddy-disease-classification/train_images')

# Define transformations
transform = transforms.Compose([
    transforms.Resize((480, 480)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),  # Added data augmentation
    transforms.RandomRotation(15),      # Added data augmentation
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1), # Added data augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Validation transform without augmentation
val_transform = transforms.Compose([
    transforms.Resize((480, 480)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create datasets using ImageFolder
train_dataset = datasets.ImageFolder(
    root=train_path,
    transform=transform
)

# Split into train and validation sets
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# Update validation transform
val_dataset.dataset.transform = val_transform

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
valid_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

# Get number of classes from the dataset
num_classes = len(train_dataset.dataset.classes)
class_names = train_dataset.dataset.classes

# Define the model using MobileNetV4 from timm
model = timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k', pretrained=True)
# Get number of features in the last layer
num_features = model.classifier.in_features
# Modify the classifier to match your number of classes
model.classifier = nn.Linear(num_features, num_classes)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training and validation functions
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    return train_loss, train_acc

def validate(model, valid_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_pred = []
    all_true = []

    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_pred.extend(predicted.cpu().numpy())
            all_true.extend(labels.cpu().numpy())

    valid_loss = running_loss / len(valid_loader)
    valid_acc = correct / total
    return valid_loss, valid_acc, all_true, all_pred

# Training loop
n_epochs = 50

for epoch in range(n_epochs):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    valid_loss, valid_acc, all_true, all_pred = validate(model, valid_loader, criterion, device)

    # Log metrics to wandb
    wandb.log({
        "Epoch": epoch + 1,
        "Train Loss": train_loss,
        "Train Accuracy": train_acc,
        "Valid Loss": valid_loss,
        "Valid Accuracy": valid_acc
    })

    print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}")

# Evaluation
all_true, all_pred = validate(model, valid_loader, criterion, device)[2:]
acc = accuracy_score(all_true, all_pred)
print("MobileNetV4 Model Accuracy on Validation Set: {:.2f}%".format(acc * 100))

cls_report = classification_report(all_true, all_pred, target_names=class_names, digits=5)
print(cls_report)

# Save the model
torch.save({
    'epoch': n_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_loss,
    'valid_loss': valid_loss,
    'class_names': class_names
}, '/kaggle/working/mobilenetv4_model.pth')

wandb.save('mobilenetv4_model.pth')
wandb.finish()