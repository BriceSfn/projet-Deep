import matplotlib
matplotlib.use("Agg")

import torch
from torch.utils.data import DataLoader

from src import (
    UNet, BCEDiceLoss, SyntheticShapes,
    fit, evaluate, set_seed, get_device, show_predictions,
)


def main():
    set_seed(0)
    device = get_device()
    print("device:", device)

    img_size = 64
    train_ds = SyntheticShapes(n_samples=64, img_size=img_size, seed=0)
    val_ds = SyntheticShapes(n_samples=24, img_size=img_size, seed=1)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    model = UNet(in_channels=3, num_classes=1, base=8).to(device)
    x, y = next(iter(train_loader))
    out = model(x.to(device))
    assert out.shape == y.shape, f"shape mismatch: {out.shape} vs {y.shape}"
    print("output shape ok:", tuple(out.shape))

    loss_fn = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    before = evaluate(model, val_loader, loss_fn, device)
    history = fit(model, train_loader, val_loader, loss_fn, optimizer, device, epochs=4, patience=5)
    after = evaluate(model, val_loader, loss_fn, device)

    print(f"val loss: {before['loss']:.4f} -> {after['loss']:.4f}")
    print(f"val dice: {before['dice']:.4f} -> {after['dice']:.4f}")
    assert after["loss"] < before["loss"], "training did not reduce the loss"

    fig = show_predictions(model, val_ds, device, n=3)
    fig.savefig("smoke_predictions.png", dpi=110, bbox_inches="tight")
    print("saved sample predictions to smoke_predictions.png")
    print("smoke test passed")


if __name__ == "__main__":
    main()
