import numpy as np
import torch
from torch import nn
import argparse
import os
import sys
from torch.utils.data import DataLoader
from utils.util import *
import torch.nn.functional as F
from datasetcClass import classDataset
from models.hrnet import hrnet

from utils.ranger import Ranger
from utils.lr_scheduler import polylr
from utils.metrics import get_metrics_values


def train(epoch, n_epochs, model, data_loader, criterion, optimizer, scheduler=None):
    """
    Train function of the model


    Parameters:
        epoch (int): Current running epoch
        n_epochs (int): Total number of epoch
        model: Model
        data_loader: Pytorch DataLoader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: lr scheduler, should support step function
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

        if scheduler:
            scheduler.step()

        losses.update(loss.item())
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d / %d] [Loss: %f]"
            % (epoch, n_epochs, i, len(data_loader), losses.avg)
        )

    print("Epoch Loss", losses.avg)


def reduce_loss(loss, reduction="mean"):
    return (
        loss.mean()
        if reduction == "mean"
        else loss.sum()
        if reduction == "sum"
        else loss
    )


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, ε: float = 0.1, reduction="mean"):
        super().__init__()
        self.ε, self.reduction = ε, reduction

    def forward(self, output, target):
        # number of classes
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        # (1-ε)* H(q,p) + ε*H(u,p)
        return (1 - self.ε) * nll + self.ε * (loss / c)


def validate_model(
    epoch, model, data_loader, criterion, best_val_iou, n_classes, output_dir
):
    """
    Validate model using Intersection over Union score

    Parameters:
        epoch (int): Current running epoch
        model: Model
        data_loader: Pytorch DataLoader
        criterion: Loss function
        best_val_iou (float): Current best value of IoU
        n_classes (int): Number of classes
        output_dir (str): Path to save weights

    Returns:
        float: Validation score

    """

    model.eval()
    losses = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    with torch.no_grad():
        for i, sample in enumerate(data_loader):
            inputs = sample["image"].type(torch.FloatTensor)
            targets = sample["label"].type(torch.LongTensor)
            if torch.cuda.is_available():
                targets = targets.cuda()
                inputs = inputs.cuda()

            outputs = model(inputs, (targets.shape[1], targets.shape[2]))["output"]
            if (
                outputs.shape[1] != targets.shape[1]
                or outputs.shape[2] != targets.shape[2]
            ):
                outputs = F.upsample(
                    input=outputs,
                    size=(targets.shape[1], targets.shape[2]),
                    mode="bilinear",
                )

            loss = criterion(outputs, targets)

            inters, uni = get_metrics_values(targets, [outputs], n_classes)
            intersection_meter.update(inters)
            union_meter.update(uni)

            losses.update(loss.item())
            sys.stdout.write(
                "\r[Epoch %d] [Batch %d / %d]" % (epoch, i, len(data_loader))
            )

    iou = intersection_meter.sum / union_meter.sum

    if best_val_iou < np.mean(iou):
        best_val_iou = np.mean(iou)
        torch.save(
            {"state_dict": model.state_dict(), "iou": np.mean(iou), "epoch": epoch},
            os.path.join(output_dir, "best.pth"),
        )
    print("")
    for i, _iou in enumerate(iou):
        print("class [{}], IoU: {:.4f}".format(i, _iou))

    print("Epoch Loss", losses.avg, "Validation_meanIou", np.mean(iou))
    return best_val_iou


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
        help="number of classes",
    )

    parser.add_argument(
        "--n_epoch",
        type=int,
        default=200,
        help="number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="batch size",
    )

    args = parser.parse_args()

    validate = False
    os.makedirs(args.output_dir, exist_ok=True)
    if os.path.exists(os.path.join(args.csvpath, "valid.csv")):
        validate = True

    # hyperparameters
    initial_learning_rate = 0.01
    weight_decay = 0.0001

    train_object = classDataset(args.csvpath, "train")
    train_loader = DataLoader(
        train_object,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
    )

    if validate:
        valid_object = classDataset(args.csvpath, "valid")
        valid_loader = DataLoader(
            valid_object,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
        )

    model = hrnet(args.n_classes)
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([0.05, 1])).cuda()

    optimizer = Ranger(
        model.parameters(), lr=initial_learning_rate, weight_decay=weight_decay
    )

    lr_scheduler = polylr(
        optimizer, args.n_epoch * len(train_loader), initial_learning_rate
    )

    best_val_iou = 0

    if torch.cuda.is_available():
        model.cuda()

    for epoch in range(1, args.n_epoch + 1):
        train(
            epoch, args.n_epoch, model, train_loader, criterion, optimizer, lr_scheduler
        )

        torch.save(
            {
                "state_dict": model.state_dict(),
                "epoch": epoch,
            },
            os.path.join(args.output_dir, "epoch_{}.pth".format(epoch)),
        )

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
