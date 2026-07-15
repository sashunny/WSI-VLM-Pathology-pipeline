"""
Fine-tune a ResNet18 on PathMNIST (public histopathology patch dataset,
9 tissue classes: adipose, background, debris, lymphocytes, mucus, smooth muscle,
normal colon mucosa, cancer-associated stroma, colorectal adenocarcinoma epithelium).

This is to train a patch-level classifier in a real WSI pipeline, using a
dataset that's small enough to fine-tune in minutes on a single GPU.

Outputs: outputs/classifier.pt, outputs/training_curve.png, printed metrics
"""
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, roc_auc_score
import medmnist
from medmnist import PathMNIST
import matplotlib.pyplot as plt

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 9
BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-4

CLASS_NAMES = [
    "adipose", "background", "debris", "lymphocytes", "mucus",
    "smooth_muscle", "normal_colon_mucosa", "cancer_associated_stroma",
    "colorectal_adenocarcinoma_epithelium",
]


def get_loaders():
    train_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    # deafulting image size to 28x28 instead of 224x224
    train_ds = PathMNIST(split="train", download=True, transform=train_tf)
    val_ds = PathMNIST(split="val", download=True, transform=eval_tf)
    test_ds = PathMNIST(split="test", download=True, transform=eval_tf)
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
        DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4),
        DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4),
    )


def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(DEVICE)


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []

    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.squeeze().long().to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            probs = torch.softmax(logits, dim=1)
            all_preds.append(probs.argmax(1).cpu().numpy())
            all_labels.append(y.cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    probs = np.concatenate(all_probs)
    acc = accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr")
    except ValueError:
        auc = float("nan")
    return total_loss / len(loader.dataset), acc, auc


def main():
    train_loader, val_loader, test_loader = get_loaders()
    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    for epoch in range(EPOCHS):
        train_loss, train_acc, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_auc = run_epoch(model, val_loader, criterion)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch+1}/{EPOCHS} | train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.4f}")

    test_loss, test_acc, test_auc = run_epoch(model, test_loader, criterion)
    print(f"\nFinal TEST metrics -> loss={test_loss:.4f} acc={test_acc:.4f} auc={test_auc:.4f}")

    torch.save({"model_state": model.state_dict(), "classes": CLASS_NAMES},
               os.path.join(OUT_DIR, "classifier.pt"))
    print(f"Saved model to {OUT_DIR}/classifier.pt")

    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.plot(history["val_acc"], label="val_acc")
    plt.xlabel("epoch")
    plt.legend()
    plt.title("Training curve")
    plt.savefig(os.path.join(OUT_DIR, "training_curve.png"))
    print(f"Saved {OUT_DIR}/training_curve.png")


if __name__ == "__main__":
    main()
