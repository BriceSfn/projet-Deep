import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src import (
    UNet, BCEDiceLoss, CocoBinaryDataset, SyntheticShapes,
    fit, evaluate, set_seed, get_device, plot_history,
)


def build_datasets(args):
    if args.train_images and args.train_ann:
        print("using coco dataset")
        train_ds = CocoBinaryDataset(args.train_images, args.train_ann,
                                     img_size=args.img_size, max_samples=args.max_train)
        val_ds = CocoBinaryDataset(args.val_images, args.val_ann,
                                   img_size=args.img_size, max_samples=args.max_val)
    else:
        print("no coco path given, using synthetic dataset")
        train_ds = SyntheticShapes(n_samples=args.max_train or 256,
                                   img_size=args.img_size, seed=0)
        val_ds = SyntheticShapes(n_samples=args.max_val or 64,
                                 img_size=args.img_size, seed=1)
    return train_ds, val_ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-images", type=str, default=None)
    parser.add_argument("--train-ann", type=str, default=None)
    parser.add_argument("--val-images", type=str, default=None)
    parser.add_argument("--val-ann", type=str, default=None)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base", type=int, default=64)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--out", type=str, default="checkpoints")
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    print("device:", device)

    train_ds, val_ds = build_datasets(args)
    print(f"train: {len(train_ds)} images | val: {len(val_ds)} images")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = UNet(in_channels=3, num_classes=1, base=args.base).to(device)
    loss_fn = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    history = fit(model, train_loader, val_loader, loss_fn, optimizer, device,
                  epochs=args.epochs, patience=args.patience, scheduler=scheduler)

    final = evaluate(model, val_loader, loss_fn, device)
    print("final validation:", {k: round(v, 4) for k, v in final.items()})

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "unet.pth")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    try:
        import matplotlib.pyplot as plt
        plot_history(history)
        plt.savefig(out_dir / "history.png", dpi=120, bbox_inches="tight")
    except Exception as exc:
        print("could not save curves:", exc)
    print("saved model and history to", out_dir.resolve())


if __name__ == "__main__":
    main()
