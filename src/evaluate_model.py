from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DATA_DIR = Path("data/processed")
MODEL_PATH = Path("models/baseline_resnet18.pt")
REPORT_DIR = Path("reports/knowledge")
FIGURE_DIR = Path("reports/figures")

BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def plot_confusion_matrix(cm, class_names, output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    print(f"Using device: {DEVICE}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Could not find model: {MODEL_PATH}")

    if not (DATA_DIR / "test").exists():
        raise FileNotFoundError(f"Could not find test folder: {DATA_DIR / 'test'}")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    class_names = checkpoint["class_names"]
    image_size = checkpoint.get("image_size", 224)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)

            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    accuracy = accuracy_score(y_true, y_pred)

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    with open(REPORT_DIR / "classification_report.json", "w") as f:
        json.dump(report_dict, f, indent=2)

    with open(REPORT_DIR / "classification_report.txt", "w") as f:
        f.write(report_text)

    with open(REPORT_DIR / "test_metrics.json", "w") as f:
        json.dump({"test_accuracy": accuracy}, f, indent=2)

    np.savetxt(
        REPORT_DIR / "confusion_matrix.csv",
        cm,
        delimiter=",",
        fmt="%d",
    )

    plot_confusion_matrix(
        cm,
        class_names,
        FIGURE_DIR / "confusion_matrix.png",
    )

    print()
    print(f"Test accuracy: {accuracy:.4f}")
    print()
    print(report_text)
    print(f"Saved: {REPORT_DIR / 'classification_report.json'}")
    print(f"Saved: {REPORT_DIR / 'classification_report.txt'}")
    print(f"Saved: {REPORT_DIR / 'test_metrics.json'}")
    print(f"Saved: {REPORT_DIR / 'confusion_matrix.csv'}")
    print(f"Saved: {FIGURE_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()