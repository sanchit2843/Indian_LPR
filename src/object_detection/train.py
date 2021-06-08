from model.fcos import FCOSDetector
import torch
from dataloader.custom_dataset import YoloDataset
import math, time
from utils.ranger import Ranger
from torch import nn
import os
import argparse
from eval import validate_one_epoch
from utils.utils import lr_func


def fit_one_epoch(epoch, model, train_loader, optimizer, steps_per_epoch, lr_params):
    TOTAL_STEPS, WARMPUP_STEPS, LR_INIT, LR_END = lr_params

    for epoch_step, data in enumerate(train_loader):
        GLOBAL_STEPS = epoch * steps_per_epoch + epoch_step + 1
        batch_imgs, batch_boxes, batch_classes, _ = data
        batch_imgs = batch_imgs.cuda()

        batch_boxes = batch_boxes.cuda()
        batch_classes = batch_classes.cuda()
        lr_params = (GLOBAL_STEPS, TOTAL_STEPS, WARMPUP_STEPS, LR_INIT, LR_END)
        lr = lr_func(*lr_params)
        for param in optimizer.param_groups:
            param["lr"] = lr
        start_time = time.time()

        model.zero_grad()
        losses = model([batch_imgs, batch_boxes, batch_classes])
        loss = losses[-1]
        loss.backward()
        optimizer.step()

        end_time = time.time()

        cost_time = int((end_time - start_time) * 1000)

        if epoch_step % 1 == 0:
            print(
                "global_steps:%d epoch:%d steps:%d/%d cls_loss:%.4f cnt_loss:%.4f reg_loss:%.4f total_loss:%.4f cost_time:%dms lr=%.4e"
                % (
                    GLOBAL_STEPS,
                    epoch + 1,
                    epoch_step + 1,
                    steps_per_epoch,
                    losses[0],
                    losses[1],
                    losses[2],
                    losses[3],
                    cost_time,
                    lr,
                )
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_txt",
        type=str,
        required=True,
        help="Location to data txt file",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="output directory to save model",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="output directory to save model",
    )
    parser.add_argument(
        "--warmupratio",
        type=float,
        default=0.12,
        help="output directory to save model",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="output directory to save model",
    )

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    # initialize dataloaders
    train_dataset = YoloDataset(args.train_txt)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=2,
    )

    model = FCOSDetector(mode="training").cuda()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    LR_INIT = 5e-4
    LR_END = 1e-6
    optimizer = Ranger(model.parameters(), lr=LR_INIT, weight_decay=1e-4)
    # check lr scheduling
    steps_per_epoch = len(train_dataset) // args.batch_size
    TOTAL_STEPS = steps_per_epoch * args.epochs
    WARMPUP_STEPS = TOTAL_STEPS * args.warmupratio

    model.train()

    lr_params = (TOTAL_STEPS, WARMPUP_STEPS, LR_INIT, LR_END)
    for epoch in range(args.epochs):
        fit_one_epoch(
            epoch,
            model,
            train_loader,
            optimizer,
            steps_per_epoch,
            lr_params,
        )
        torch.save(
            model.state_dict(),
            "{}/hrnet18v2_ranger{}.pth".format(args.output_path, epoch + 1),
        )


if __name__ == "__main__":
    main()