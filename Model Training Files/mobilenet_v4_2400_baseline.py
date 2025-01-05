# Importing Libraries
import os
import random
import numpy as np
import pandas as pd
import wandb
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from pathlib import Path
import timm
import math

# Setting Seeds for Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Hyperparameters and Configurations
LEARNING_RATE = 0.0001
ARCHITECTURE = "mobilenet_v4_2400_baseline"
EPOCHS = 75
BATCH_SIZE = 64
OPTIMISER = "Adam"
LOSS_FUNCTION = "CrossEntropyLoss"
NUM_CLASSES = 10
PRETRAINED = True

trainPath = '/kaggle/input/paddy-disease-classification/train_images'
testPath = '/kaggle/input/paddy-disease-classification/test_images'
train_labels_path = '/kaggle/input/paddy-disease-classification/train.csv'

augment = False
attention = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")

# Initialize wandb
wandb.login(key='3043bbf9bb6b869c873214e9e89a66b9894dcaa2')
wandb.init(project="rice-disease-classification-mobilenet_v4",
           name="MobileNet v4 Baseline (2400)",
           config={
               "learning_rate": LEARNING_RATE,
               "architecture": ARCHITECTURE,
               "epochs": EPOCHS,
               "batch_size": BATCH_SIZE,
               "optimizer": OPTIMISER,
               "loss_function": LOSS_FUNCTION,
                "pretrained": PRETRAINED,
                "num_classes": NUM_CLASSES
           })

# Efficient Attention Mechanisms
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.mean((2, 3))
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y

class ECABlock(nn.Module):
    def __init__(self, in_channels, gamma=2, b=1):
        super(ECABlock, self).__init__()
        kernel_size = int(abs((math.log2(in_channels) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(logits, targets)
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return loss.mean()

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.classes = classes

    def forward(self, logits, targets):
        confidence = 1.0 - self.smoothing
        smooth_value = self.smoothing / self.classes
        one_hot = torch.zeros_like(logits).scatter(1, targets.unsqueeze(1), 1)
        smoothed_targets = one_hot * confidence + smooth_value
        loss = -smoothed_targets * torch.log_softmax(logits, dim=-1)
        return loss.sum(dim=1).mean()
    
class FocalLabelSmoothingComboLoss(nn.Module):
    def __init__(self, focal_weight=0.5, smoothing_weight=0.5, focal_params=None, smoothing_params=None):
        super(FocalLabelSmoothingComboLoss, self).__init__()
        self.focal_weight = focal_weight
        self.smoothing_weight = smoothing_weight
        self.focal_loss = FocalLoss(**(focal_params or {}))
        self.smoothing_loss = LabelSmoothingLoss(**(smoothing_params or {}))

    def forward(self, logits, targets):
        focal_loss = self.focal_loss(logits, targets)
        smoothing_loss = self.smoothing_loss(logits, targets)
        return self.focal_weight * focal_loss + self.smoothing_weight * smoothing_loss

# Dataset Class
class ImageDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        disease = self.df.iloc[idx]['label']
        img_name = self.df.iloc[idx]['image_id']
        img_path = os.path.join(self.img_dir, disease, img_name)
        image = Image.open(img_path)
        label = self.df.iloc[idx]['label_encoded']

        if self.transform:
            image = self.transform(image)

        return image, label
    
def prepare_data():
    train_path = Path(trainPath)
    test_path = Path(testPath)

    # Loading Train Labels
    train_df = pd.read_csv(train_labels_path)
    label_encoder = LabelEncoder()
    train_df['label_encoded'] = label_encoder.fit_transform(train_df['label'])

    # Augmentations
    if augment == True:
        augmentations = transforms.Compose([
            transforms.Resize((480, 480)),
            transforms.RandomChoice([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        augmentations = transforms.Compose([
            transforms.Resize((480, 480)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((480, 480)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['label'])

    train_dataset = ImageDataset(df=train_df, img_dir=train_path, transform=augmentations)
    valid_dataset = ImageDataset(df=valid_df, img_dir=train_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

    return train_loader, valid_loader, label_encoder, len(np.unique(train_df['label_encoded']))

# Training and Validation
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progress Bar for Training
    pbar = tqdm(train_loader, desc='Train', leave=False, unit="batch")
    for images, labels in pbar:
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
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}', 
            'Accuracy': f'{(predicted == labels).sum().item() / labels.size(0):.4f}'
        })

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
    
    # Progress Bar for Validation
    pbar = tqdm(valid_loader, desc='Validation', leave=False, unit="batch")
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_pred.extend(predicted.cpu().numpy())
            all_true.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}', 
                'Accuracy': f'{(predicted == labels).sum().item() / labels.size(0):.4f}'
            })

    valid_loss = running_loss / len(valid_loader)
    valid_acc = correct / total
    return valid_loss, valid_acc, all_true, all_pred

# Define Modified Model
def LARSMobileNetv4():
    # Load Pretrained Model
    model = timm.create_model("hf_hub:timm/mobilenetv4_conv_small.e2400_r224_in1k", pretrained=True, num_classes=NUM_CLASSES)
    if attention == True:
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d): # Add SE and ECA Blocks to Conv2d layers
                se_block = SEBlock(module.out_channels)
                eca_block = ECABlock(module.out_channels)
                module.add_module("se", se_block)
                module.add_module("eca", eca_block)
        return model
    else:
        return model

# Main Function
def main():
    print('Starting Data Preparation')
    train_loader, valid_loader, label_encoder, num_classes = prepare_data()
    print('Data Preparation Completed')

    # Initializing Model
    print('Model Initialization')
    model = LARSMobileNetv4().to(DEVICE)

    # Print Param Count
    print(f"Number of parameters in the model: {sum(p.numel() for p in model.parameters())}")

    # criterion = FocalLabelSmoothingComboLoss(smoothing_params={"classes": NUM_CLASSES})
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Storing Best Validation Score as Checkpoints
    best_valid_acc = 0.0
    best_valid_loss = float('inf')
    save_dir = Path('model_checkpoints')
    save_dir.mkdir(exist_ok=True)

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        # Training Phase
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, DEVICE)
        print(f"Training: Loss = {train_loss:.4f}, Accuracy = {train_acc:.4f}")

        # Validation Phase
        valid_loss, valid_acc, all_true, all_pred = validate(model, valid_loader, criterion, DEVICE)
        print(f"Validation: Loss = {valid_loss:.4f}, Accuracy = {valid_acc:.4f}")

        # Save Best Validation Accuracy Model
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model_path = save_dir / f'{ARCHITECTURE}_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'train_acc': train_acc,
                'valid_acc': valid_acc,
            }, best_model_path)
            print(f"Best Validation Model Saved! \nValidation Accuracy: {valid_acc:.4f}")
            wandb.save(str(best_model_path))

        wandb.log({
            "Epoch": epoch + 1,
            "Train Loss": train_loss,
            "Train Accuracy": train_acc,
            "Valid Loss": valid_loss,
            "Valid Accuracy": valid_acc
        })

    # Final Evaluation
    class_names = label_encoder.classes_
    all_true, all_pred = validate(model, valid_loader, criterion, DEVICE)[2:]
    acc = accuracy_score(all_true, all_pred)
    print("\nEfficient MobileNetV4 Model Accuracy on Validation Set: {:.2f}%".format(acc * 100))
    cls_report = classification_report(all_true, all_pred, target_names=class_names, digits=5)
    print(cls_report)

    torch.save(model.state_dict(), f'{ARCHITECTURE}_latest_model.pth')
    wandb.save(f'{ARCHITECTURE}_latest_model.pth')

if __name__ == '__main__':
    main()
    wandb.finish()
