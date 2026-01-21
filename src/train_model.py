import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from dataset_loader import ViolenceDataset


# --------- Model (CNN + LSTM) ----------
class CNN_LSTM(nn.Module):
    def __init__(self, hidden_size=256, num_classes=2):
        super(CNN_LSTM, self).__init__()
        # Use ResNet18 as feature extractor
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(base_model.children())[:-1])  # remove FC layer
        self.feature_dim = base_model.fc.in_features

        # ✅ Unfreeze CNN (train the full network)
        for param in self.cnn.parameters():
            param.requires_grad = True

        # LSTM
        self.lstm = nn.LSTM(self.feature_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)       # merge batch & time
        feats = self.cnn(x)              # (B*T, 512, 1, 1)
        feats = feats.view(B, T, -1)     # (B, T, 512)

        out, _ = self.lstm(feats)        # (B, T, hidden_size)
        out = out[:, -1, :]              # take last timestep
        out = self.fc(out)               # (B, num_classes)
        return out


# --------- Training Function ----------
def train_model():
    # ✅ Stronger data augmentation
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(112, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor()
    ])

    # datasets
    train_dataset = ViolenceDataset("data/train.csv", num_frames=16, transform=transform)
    val_dataset = ViolenceDataset("data/val.csv", num_frames=16, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_LSTM(hidden_size=256, num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # training setup
    epochs = 20
    best_val_acc = 0.0

    # ✅ log dictionary for plotting later
    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss, total_correct = 0, 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        for frames, labels in loop:
            frames, labels = frames.to(device), labels.to(device)
            outputs = model(frames)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += (outputs.argmax(1) == labels).sum().item()

            loop.set_postfix(loss=loss.item())

        acc = total_correct / len(train_dataset)
        print(f"\nTrain Loss: {total_loss:.4f}, Train Acc: {acc:.4f}")

        # validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for frames, labels in val_loader:
                frames, labels = frames.to(device), labels.to(device)
                outputs = model(frames)
                correct += (outputs.argmax(1) == labels).sum().item()
        val_acc = correct / len(val_dataset)
        print(f"Validation Acc: {val_acc:.4f}")

        # ✅ save metrics for plotting
        history["train_loss"].append(total_loss)
        history["train_acc"].append(acc)
        history["val_acc"].append(val_acc)

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/cnn_lstm_best.pth")
            print(f"✅ Best model saved with val_acc={val_acc:.4f}")

    # ✅ Save log after training
    with open("training_log.json", "w") as f:
        json.dump(history, f)
    print("📊 Training log saved to training_log.json")

    print("Training finished.")


if __name__ == "__main__":
    train_model()
