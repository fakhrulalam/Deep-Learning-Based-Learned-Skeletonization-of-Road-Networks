import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

# ---------------------
# Dataset Loader
# ---------------------
class RoadDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        self.input_paths = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)])
        self.target_paths = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir)])

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        x = cv2.imread(self.input_paths[idx], cv2.IMREAD_GRAYSCALE) / 255.0
        y = cv2.imread(self.target_paths[idx], cv2.IMREAD_GRAYSCALE) / 255.0
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        return x, y

# ---------------------
# U-Net Model (Simple)
# ---------------------
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 8, 2, stride=2), nn.ReLU(),
            nn.Conv2d(8, 1, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return self.sigmoid(x)

# ---------------------
# Dice Loss Definition
# ---------------------
class DiceLoss(nn.Module):
    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        return 1 - ((2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth))

# ---------------------
# Training Loop
# ---------------------
def train():
    input_dir = "data/input"
    target_dir = "data/target"
    dataset = RoadDataset(input_dir, target_dir)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = UNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    bce = nn.BCELoss()
    dice = DiceLoss()
    loss_fn = lambda pred, target: 0.5 * bce(pred, target) + 0.5 * dice(pred, target)

    os.makedirs("results", exist_ok=True)
    loss_history = []

    for epoch in range(30):
        model.train()
        total_loss = 0
        for x, y in dataloader:
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "results/model.pth")

    # Save loss curve
    plt.plot(loss_history)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("results/loss_curve.png")
    plt.close()

    # Save predictions for first 3 samples
    model.eval()
    with torch.no_grad():
        for i in range(3):
            x, y = dataset[i]
            pred = model(x.unsqueeze(0)).squeeze().numpy()

            # Soft prediction visualization
            fig, axs = plt.subplots(1, 3, figsize=(10, 3))
            axs[0].imshow(x.squeeze(), cmap='gray')
            axs[0].set_title("Input")
            axs[1].imshow(y.squeeze(), cmap='gray')
            axs[1].set_title("Ground Truth")
            axs[2].imshow(pred, cmap='gray')
            axs[2].set_title("Prediction (Soft)")
            for ax in axs: ax.axis('off')
            plt.tight_layout()
            plt.savefig(f"results/preview_{i:05d}.png")
            plt.close()

            # Binarize + skeletonize
            pred_bin = (pred > 0.2).astype(np.uint8)  # lower threshold
            pred_skel = skeletonize(pred_bin).astype(np.uint8) * 255
            cv2.imwrite(f"results/sample_{i:05d}.png", pred_skel)

if __name__ == "__main__":
    train()
