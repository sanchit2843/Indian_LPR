import numpy as np
import torch
from torch import nn
import argparse
import os
import sys
from torch.utils.data import DataLoader
import torch.optim as optim
from utils.util import *
import torch.nn.functional as F
from datasetcClass import classDataset
from models.hrnet import hrnet
from evaluation import validate_model
from ranger import Ranger


def train(epoch, n_epochs, model, data_loader, criterion, optimizer):
    """
    Train function of the model


    Parameters:
        epoch (int): Current running epoch
        n_epochs (int): Total number of epoch
        model: Model
        data_loader: Pytorch DataLoader
        criterion: Loss function
        optimizer: Optimizer

    """

    model.train()
    losses = AverageMeter()

    for i, sample in enumerate(data_loader):
        inputs = sample["image"]
        targets = sample["label"].long()
        if torch.cuda.is_available():
            targets = targets.cuda()
            inputs = inputs.cuda()

        model.zero_grad()
        outputs = model(inputs)["output"]
        if outputs.shape[1] != targets.shape[1] or outputs.shape[2] != targets.shape[2]:
            outputs = F.upsample(
                input=outputs,
                size=(targets.shape[1], targets.shape[2]),
                mode="bilinear",
            )
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        losses.update(loss.item())
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d / %d] [Loss: %f]"
            % (epoch, n_epochs, i, len(data_loader), losses.avg)
        )
    print("Epoch Loss", losses.avg)


def main():
    ####################argument parser#################################
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csvpath",
        type=str,
        required=True,
        default="./",
        help="Location to data csv file",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="output directory to save model",
    )

    parser.add_argument(
        "--n_classes",
        type=int,
        default=2,
        help="number of claesses",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="height of image",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="width of image",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="batch size",
    )

    args = parser.parse_args()

    validate = True

    if os.path.exists(os.path.join(args.csvpath, "valid.csv")):
        validate = True

    train_object = classDataset(args.csvpath, "train", args.height, args.width)
    train_loader = DataLoader(
        train_object,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
    )

    if validate:
        valid_object = classDataset(args.csvpath, "valid", args.height, args.width)
        valid_loader = DataLoader(
            valid_object,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
        )

    model = hrnet(args.n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = Ranger(model.parameters(), lr=5e-3)
    best_val_iou = 0

    if torch.cuda.is_available():
        model.cuda()

    start_epoch = 1
    n_epochs = 200
    for epoch in range(start_epoch, n_epochs + 1):
        train(epoch, n_epochs, model, train_loader, criterion, optimizer)
        if validate:
            best_val_iou = validate_model(
                epoch,
                model,
                valid_loader,
                criterion,
                best_val_iou,
                args.n_classes,
                args.output_dir,
            )


if __name__ == "__main__":
    main()


# python training.py --csvpath dataset/steel
