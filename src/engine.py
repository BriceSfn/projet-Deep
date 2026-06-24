import copy

import torch
from tqdm.auto import tqdm

from .metrics import dice_coeff, iou_score, pixel_accuracy


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    running = 0.0
    for images, masks in tqdm(loader, leave=False, desc="train"):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()
        running += loss.item() * images.size(0)
    return running / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    loss_sum = dice_sum = iou_sum = acc_sum = 0.0
    for images, masks in tqdm(loader, leave=False, desc="val"):
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        bs = images.size(0)
        loss_sum += loss_fn(logits, masks).item() * bs
        dice_sum += dice_coeff(logits, masks) * bs
        iou_sum += iou_score(logits, masks) * bs
        acc_sum += pixel_accuracy(logits, masks) * bs
    n = len(loader.dataset)
    return {"loss": loss_sum / n, "dice": dice_sum / n, "iou": iou_sum / n, "acc": acc_sum / n}


def fit(model, train_loader, val_loader, loss_fn, optimizer, device,
        epochs=20, patience=5, scheduler=None, verbose=True):
    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": [], "val_acc": []}
    best_val = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    epochs_without_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val = evaluate(model, val_loader, loss_fn, device)
        if scheduler is not None:
            scheduler.step(val["loss"])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["val_dice"].append(val["dice"])
        history["val_iou"].append(val["iou"])
        history["val_acc"].append(val["acc"])

        if verbose:
            print(
                f"epoch {epoch:3d}/{epochs} | train loss {train_loss:.4f} | "
                f"val loss {val['loss']:.4f} | dice {val['dice']:.4f} | "
                f"iou {val['iou']:.4f} | acc {val['acc']:.4f}"
            )

        if val["loss"] < best_val - 1e-4:
            best_val = val["loss"]
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                if verbose:
                    print(f"early stopping at epoch {epoch} (best val loss {best_val:.4f})")
                break

    model.load_state_dict(best_weights)
    return history
