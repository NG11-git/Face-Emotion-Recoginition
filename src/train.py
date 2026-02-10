import os 
import torch
from torchvision import transforms
from torchvision import datasets 
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import torch.optim as optim
from utils import EarlyStopping, plot_curves

# transformations for the training and validation sets
image_transform = {
    'train' : transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8,1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        )
    ]),
    'valid' : transforms.Compose([
        transforms.Resize(225),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        )
    ])
}

# number classes in data
num_classes = 7

# batch size
bs = 64

valid_dir = 'validation'
train_dir = 'train'


# Load the dataset
train_dataset = datasets.ImageFolder(root=train_dir, transform=image_transform['train'])
valid_dataset = datasets.ImageFolder(root=valid_dir, transform=image_transform['valid'])

# size of the data used for accuracy and validation
valid_size = len(valid_dataset)
train_size = len(train_dataset)

# iterator for the data using dataloader

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=bs, shuffle=True)

# print the size
print(train_size, valid_size)

#----------------------------------#

# We'll use CBAM(Convolutional Block Attention Module), which combine
# Channel Attention : Learns which feature maps are important
# Spatial Attetion : Learns where to focus within the feature maps

# Define CBAM Module

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1,bias=False)
        )
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return torch.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat   = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(x_cat))

class CBAM(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x
    
# Define ResNet18 with CBAM

class ResNet18_CBAM(nn.Module):
    def __init__(self, num_classes=7):       #FER has 7 classes
        super().__init__()
        base_model = resnet18(pretrained=True)
        self.features = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            CBAM(64),
            base_model.layer2,
            CBAM(128),
            base_model.layer3,
            CBAM(256),
            base_model.layer4,
            CBAM(512),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)
#----------------------------------#

# device configuration
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model, loss function, optimizer, scheduler and early stopping

epochs = 50
bs  = 64
lr = 1e-4
patience = 7
num_classes = 7
model = ResNet18_CBAM(num_classes=7).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) 
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
early_stopper = EarlyStopping(patience=patience, path='best_model.pt')

# training and validation loop


#----training loop-----
train_losses, val_losses = [], []
for epoch in range(epochs):
    model.train()
    running_loss=0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    #----Validation----
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += labels.size(0)
            total += labels.size(0)

    avg_val_loss = val_loss / len(valid_loader)
    val_losses.append(avg_val_loss)
    val_acc = correct / total

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss: .4f} | Val Loss:{avg_val_loss:.4f} | Val Acc:{val_acc:.4f}")
    scheduler.step(avg_val_loss)
    early_stopper(avg_val_loss, model)

    if early_stopper.early_stop:
        print("Early stopping Triggered.")
        break

# plot the training and validation curves
plot_curves(train_losses, val_losses)

# load the best model
model.load_state_dict(torch.load("/kaggle/working/best_model.pt"))

# save the final model
dummy_input = torch.randn(1, 3, 48, 48)
torch.onnx.export(model.cpu(), dummy_input, "emotional_model.onnx")