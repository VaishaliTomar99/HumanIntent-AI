from torch.utils.data import DataLoader
from dataset_loader import ViolenceDataset

# use train.csv
dataset = ViolenceDataset("data/train.csv", num_frames=16)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

for frames, labels in loader:
    print("Batch frames shape:", frames.shape)  # (B, T, C, H, W)
    print("Batch labels:", labels)
    break
