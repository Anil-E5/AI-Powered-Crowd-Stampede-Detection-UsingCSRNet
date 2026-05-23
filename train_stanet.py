import os
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# -------------------------
# 1. Setup Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# 2. Dataset Preparation
# -------------------------
dataset_zip = "crowd_img.zip"
dataset_dir = "crowd_dataset"

if not os.path.exists(dataset_dir):
    with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
    print("Dataset extracted at:", dataset_dir)

train_dir = os.path.join(dataset_dir, "train")
test_dir = os.path.join(dataset_dir, "test")

class CrowdDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        # Dummy density map for now (replace with real labels if available)
        density_map = np.ones((image.size[1]//8, image.size[0]//8), dtype=np.float32)  

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(density_map).unsqueeze(0)

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = CrowdDataset(train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# -------------------------
# 3. CSRNet Model Definition
# -------------------------
# Minimal CSRNet
class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        # Frontend (VGG16-like)
        self.frontend = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # Backend
        self.backend = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)
        )

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return x

model = CSRNet().to(device)

# -------------------------
# 4. Loss and Optimizer
# -------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# -------------------------
# 5. Training Loop
# -------------------------
num_epochs = 50

for epoch in range(1, num_epochs+1):
    model.train()
    running_loss = 0.0
    for images, density_maps in train_loader:
        images = images.to(device)
        density_maps = density_maps.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, density_maps)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch}/{num_epochs}] Loss: {epoch_loss:.4f}")

# -------------------------
# 6. Save Model
# -------------------------
torch.save(model.state_dict(), "csrnet_crowd.pth")
print("🎉 Model saved as csrnet_crowd.pth")
