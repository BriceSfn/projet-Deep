import random

import numpy as np
import torch

from .dataset import MEAN, STD


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def denormalize(img_tensor):
    img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(STD) + np.array(MEAN)
    return np.clip(img, 0, 1)


def plot_history(history, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax[0].plot(epochs, history["train_loss"], label="train")
    ax[0].plot(epochs, history["val_loss"], label="val")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("epoch")
    ax[0].legend()
    ax[1].plot(epochs, history["val_dice"], label="val Dice", color="green")
    ax[1].plot(epochs, history["val_iou"], label="val IoU", color="orange")
    ax[1].set_title("Validation metrics")
    ax[1].set_xlabel("epoch")
    ax[1].legend()
    return ax


@torch.no_grad()
def show_predictions(model, dataset, device, n=4, threshold=0.5, seed=0):
    import matplotlib.pyplot as plt

    model.eval()
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(n, len(dataset)), replace=False)

    fig, axes = plt.subplots(len(indices), 3, figsize=(9, 3 * len(indices)))
    if len(indices) == 1:
        axes = axes[None, :]
    for row, idx in enumerate(indices):
        image, mask = dataset[int(idx)]
        logits = model(image[None].to(device))
        pred = (torch.sigmoid(logits)[0, 0].cpu().numpy() > threshold).astype(float)

        axes[row, 0].imshow(denormalize(image))
        axes[row, 0].set_title("image")
        axes[row, 1].imshow(mask[0], cmap="gray")
        axes[row, 1].set_title("ground truth")
        axes[row, 2].imshow(pred, cmap="gray")
        axes[row, 2].set_title("prediction")
        for c in range(3):
            axes[row, c].axis("off")
    fig.tight_layout()
    return fig
