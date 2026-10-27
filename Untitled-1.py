
import os
import copy
import random
import urllib.request
import zipfile

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# =========================================================
# 1. PROJECT SETTINGS
# =========================================================

SEED = 42
BATCH_SIZE = 8
EPOCHS = 5

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "hymenoptera_data")
ZIP_PATH = os.path.join(PROJECT_DIR, "hymenoptera_data.zip")

MODEL_PATH = os.path.join(
    PROJECT_DIR, "best_finetuned_resnet18.pth"
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# =========================================================
# 2. DOWNLOAD AND EXTRACT DATASET
# =========================================================

DATA_URL = (
    "https://download.pytorch.org/tutorial/"
    "hymenoptera_data.zip"
)

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

if not (
    os.path.isdir(TRAIN_DIR)
    and os.path.isdir(VAL_DIR)
):
    print("Downloading dataset...")

    if not os.path.isfile(ZIP_PATH):
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)

    print("Extracting dataset...")

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_file:
        zip_file.extractall(PROJECT_DIR)

    print("Dataset ready.")


# =========================================================
# 3. IMAGE TRANSFORMS
# =========================================================

weights = ResNet18_Weights.DEFAULT

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=imagenet_mean,
        std=imagenet_std
    )
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=imagenet_mean,
        std=imagenet_std
    )
])


# =========================================================
# 4. LOAD DATASET
# =========================================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transforms
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transforms
)

class_names = train_dataset.classes

if train_dataset.class_to_idx != val_dataset.class_to_idx:
    raise ValueError(
        "Training and validation classes do not match."
    )

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Classes:", class_names)
print("Class mapping:", train_dataset.class_to_idx)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# =========================================================
# 5. LOAD PRETRAINED RESNET18
# =========================================================

model = resnet18(weights=weights)

# Freeze every parameter first.
for parameter in model.parameters():
    parameter.requires_grad = False

# Unfreeze the final residual block.
for parameter in model.layer4.parameters():
    parameter.requires_grad = True

# Replace and train the final classification layer.
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, len(class_names))

model = model.to(device)

print("\nModel loaded.")
print("Classifier:", in_features, "->", len(class_names))
print("Trainable parameters:")

for name, parameter in model.named_parameters():
    if parameter.requires_grad:
        print(" ", name)


# =========================================================
# 6. LOSS FUNCTION AND OPTIMIZER
# =========================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam([
    {
        "params": model.layer4.parameters(),
        "lr": 0.0001
    },
    {
        "params": model.fc.parameters(),
        "lr": 0.001
    }
])

# Learning-rate scheduler reduces the learning rates
# when validation loss stops improving.
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=1
)


# =========================================================
# 7. TRAINING AND VALIDATION
# =========================================================

best_val_accuracy = 0.0
best_model_weights = copy.deepcopy(model.state_dict())

history = {
    "train_loss": [],
    "val_loss": [],
    "train_accuracy": [],
    "val_accuracy": []
}

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    for phase in ["train", "val"]:

        if phase == "train":
            model.train()

            # Keep frozen backbone BatchNorm statistics fixed.
            model.conv1.eval()
            model.bn1.eval()
            model.layer1.eval()
            model.layer2.eval()
            model.layer3.eval()

        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        current_loader = (
            train_loader if phase == "train" else val_loader
        )

        for images, labels in current_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):

                outputs = model(images)
                loss = criterion(outputs, labels)

                predictions = torch.argmax(outputs, dim=1)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * images.size(0)

            running_corrects += torch.sum(
                predictions == labels
            ).item()

        dataset_size = len(current_loader.dataset)

        epoch_loss = running_loss / dataset_size
        epoch_accuracy = running_corrects / dataset_size

        print(
            f"{phase.title()} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Accuracy: {epoch_accuracy * 100:.2f}%"
        )

        history[f"{phase}_loss"].append(epoch_loss)
        history[f"{phase}_accuracy"].append(epoch_accuracy)

        if phase == "val":
            scheduler.step(epoch_loss)

            if epoch_accuracy > best_val_accuracy:
                best_val_accuracy = epoch_accuracy
                best_model_weights = copy.deepcopy(
                    model.state_dict()
                )

                torch.save(
                    {
                        "model_state_dict": best_model_weights,
                        "class_names": class_names,
                        "best_val_accuracy": best_val_accuracy
                    },
                    MODEL_PATH
                )

                print("Best model saved.")


# =========================================================
# 8. RESTORE BEST MODEL
# =========================================================

model.load_state_dict(best_model_weights)
model.eval()

print("\nBest validation accuracy:",
      f"{best_val_accuracy * 100:.2f}%")


# =========================================================
# 9. PLOT LOSS
# =========================================================

epochs_range = range(1, EPOCHS + 1)

plt.figure(figsize=(8, 5))
plt.plot(
    epochs_range,
    history["train_loss"],
    marker="o",
    label="Training Loss"
)
plt.plot(
    epochs_range,
    history["val_loss"],
    marker="o",
    label="Validation Loss"
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Fine-Tuning: Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(PROJECT_DIR, "finetuning_loss.png"),
    dpi=150
)
plt.close()


# =========================================================
# 10. PLOT ACCURACY
# =========================================================

plt.figure(figsize=(8, 5))
plt.plot(
    epochs_range,
    [x * 100 for x in history["train_accuracy"]],
    marker="o",
    label="Training Accuracy"
)
plt.plot(
    epochs_range,
    [x * 100 for x in history["val_accuracy"]],
    marker="o",
    label="Validation Accuracy"
)
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Fine-Tuning: Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(PROJECT_DIR, "finetuning_accuracy.png"),
    dpi=150
)
plt.close()


# =========================================================
# 11. CONFUSION MATRIX
# =========================================================

all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in val_loader:

        images = images.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu()

        all_labels.extend(labels.numpy().tolist())
        all_predictions.extend(predictions.numpy().tolist())

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(class_names)))
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(cmap="Blues", values_format="d")
plt.title("Fine-Tuned ResNet18 Confusion Matrix")
plt.tight_layout()
plt.savefig(
    os.path.join(PROJECT_DIR, "finetuning_confusion_matrix.png"),
    dpi=150
)
plt.close()


# =========================================================
# 12. DISPLAY SAMPLE PREDICTIONS
# =========================================================

mean_tensor = torch.tensor(
    imagenet_mean
).view(3, 1, 1)

std_tensor = torch.tensor(
    imagenet_std
).view(3, 1, 1)

images, labels = next(iter(val_loader))
images_on_device = images.to(device)

with torch.no_grad():
    outputs = model(images_on_device)
    probabilities = torch.softmax(outputs, dim=1)
    predictions = outputs.argmax(dim=1).cpu()

num_images = min(8, len(images))

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.flatten()

for i in range(num_images):

    image = images[i].cpu() * std_tensor + mean_tensor
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()

    true_label = class_names[labels[i].item()]
    predicted_label = class_names[predictions[i].item()]
    confidence = probabilities[i, predictions[i]].item() * 100

    axes[i].imshow(image)
    axes[i].set_title(
        f"True: {true_label}\n"
        f"Pred: {predicted_label} ({confidence:.1f}%)"
    )
    axes[i].axis("off")

for i in range(num_images, len(axes)):
    axes[i].axis("off")

plt.suptitle("Fine-Tuned ResNet18: Sample Predictions")
plt.tight_layout()
plt.savefig(
    os.path.join(PROJECT_DIR, "finetuning_predictions.png"),
    dpi=150
)
plt.close()


# =========================================================
# 13. FINAL REPORT
# =========================================================

print("\n" + "=" * 55)
print("FINE-TUNING PROJECT COMPLETED")
print("=" * 55)
print("Classes:", class_names)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print(f"Best validation accuracy: {best_val_accuracy * 100:.2f}%")
print("Model saved:", MODEL_PATH)

print("\nGenerated files:")
print("- best_finetuned_resnet18.pth")
print("- finetuning_loss.png")
print("- finetuning_accuracy.png")
print("- finetuning_confusion_matrix.png")
print("- finetuning_predictions.png")
