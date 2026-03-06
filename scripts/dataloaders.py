import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# -------- adjust if needed --------
DATA_ROOT = r"E:/ASL_Project/dataset_split"
IMG_SIZE  = 224
BATCH_SIZE = 64
NUM_WORKERS = 4
# ----------------------------------

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
    transforms.RandomRotation(12),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def get_loaders():
    train_ds = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"),
                                    transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(DATA_ROOT, "val"),
                                    transform=val_tf)
    test_ds  = datasets.ImageFolder(os.path.join(DATA_ROOT, "test"),
                                    transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, val_loader, test_loader, train_ds.classes

if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = get_loaders()
    print("Number of classes:", len(classes))
    imgs, labels = next(iter(train_loader))
    print("Train batch shape:", imgs.shape)
    print("Labels shape:", labels.shape)
