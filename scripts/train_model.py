import os, time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm

# --------- paths & hyperparameters ---------
DATA_ROOT   = r"E:\ASL_Project\dataset_split"
IMG_SIZE    = 224
BATCH_SIZE  = 16
NUM_CLASSES = 29
EPOCHS_HEAD = 2       # phase 1
EPOCHS_FINE = 5      # phase 2
LR_HEAD     = 1e-3
LR_FINE     = 1e-4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# --------------------------------------------

def main():
    # transforms
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
        transforms.RandomRotation(12),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    # datasets & loaders
    train_ds = datasets.ImageFolder(os.path.join(DATA_ROOT,"train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(DATA_ROOT,"val"),   transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # model
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=NUM_CLASSES)
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    def train_epoch(loader, optimizer):
        model.train()
        running_loss, correct, total = 0, 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
        return running_loss/total, correct/total

    def eval_epoch(loader):
        model.eval()
        loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss += criterion(outputs, labels).item() * imgs.size(0)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
        return loss/total, correct/total

    # ---- Phase 1: train classifier head ----
    for name,param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=LR_HEAD, weight_decay=1e-4)

    print("=== Phase 1: Training classifier head ===")
    for epoch in range(EPOCHS_HEAD):
        tr_loss, tr_acc = train_epoch(train_loader, optimizer)
        val_loss, val_acc = eval_epoch(val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS_HEAD} | Train Acc: {tr_acc:.4f} | Val Acc: {val_acc:.4f}")

    # ---- Phase 2: fine-tune entire model ----
    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.AdamW(model.parameters(), lr=LR_FINE, weight_decay=1e-4)

    print("\n=== Phase 2: Fine-tuning entire model ===")
    best_val = 0.0
    for epoch in range(EPOCHS_FINE):
        tr_loss, tr_acc = train_epoch(train_loader, optimizer)
        val_loss, val_acc = eval_epoch(val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS_FINE} | Train Acc: {tr_acc:.4f} | Val Acc: {val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            os.makedirs(r"E:\ASL_Project\models", exist_ok=True)
            torch.save(model.state_dict(), r"E:\ASL_Project\models\best_efficientnet_b0.pth")
            print(f"  ✅ New best model saved (val_acc={val_acc:.4f})")

    print("Training finished. Best val accuracy:", best_val)

if __name__ == "__main__":
    main()
