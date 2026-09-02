"""
Trainiert den FishBotCNN auf den Bildern in data/<klasse>/.

Voraussetzung: Jede der 4 Klassen-Ordner (data/fisch_auf_balken,
data/fisch_links_von_balken, data/fisch_rechts_von_balken, data/balken_weg)
enthaelt Bilder.

Ausfuehren: python src/train.py
"""

import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Erlaubt Ausführung sowohl aus src/ als auch aus dem Projekt-Root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CLASSES, MODEL_INPUT_SIZE, MODEL_PATH
from model import FishBotCNN

# Projekt-Root = eine Ebene über src/, unabhängig vom Arbeitsverzeichnis
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_PATH = os.path.join(PROJECT_ROOT, MODEL_PATH)
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.Resize((MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0])),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # etwas Robustheit
        transforms.ToTensor(),
    ])

    full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

    # Sicherstellen, dass ImageFolder-Reihenfolge zu unserer CLASSES-Liste passt
    print(f"Gefundene Klassen: {full_dataset.classes}")
    assert set(full_dataset.classes) == set(CLASSES), (
        "Klassen in data/ stimmen nicht mit config.CLASSES ueberein!"
    )

    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = FishBotCNN(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total if total > 0 else 0.0

        print(f"Epoch {epoch:2d}/{EPOCHS} - train_loss: {train_loss:.4f} - val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "classes": full_dataset.classes,
            }, MODEL_PATH)
            print(f"  -> neues bestes Modell gespeichert ({val_acc:.4f})")

    print(f"\nFertig. Bestes val_acc: {best_val_acc:.4f}. Modell in {MODEL_PATH}")


if __name__ == "__main__":
    main()
