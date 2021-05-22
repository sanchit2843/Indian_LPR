from model.fcos import FCOSDetector
import torch
from dataloader.custom_dataset import YoloDataset
import math, time
from utils.ranger import Ranger
from torch import nn
import os
import argparse
from eval import validate_one_epoch


def fit_one_epoch(epoch, model, train_loader, optimizer, steps):
    GLOBAL_STEPS, steps_per_epoch = steps
    for param in optimizer.param_groups:
        lr = param["lr"]
        break
    for epoch_step, data in enumerate(train_loader):

        batch_imgs, batch_boxes, batch_classes, _ = data
        batch_imgs = batch_imgs.cuda()

        batch_boxes = batch_boxes.cuda()
        batch_classes = batch_classes.cuda()

        start_time = time.time()

        model.zero_grad()
        losses = model([batch_imgs, batch_boxes, batch_classes])
        loss = losses[-1]
        loss.backward()
        optimizer.step()

        end_time = time.time()

        cost_time = int((end_time - start_time) * 1000)

        if epoch_step % 100 == 0:
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

        GLOBAL_STEPS += 1


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
    )

    model = FCOSDetector(mode="training").cuda()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    LR_INIT = 0.01

    optimizer = Ranger(model.parameters(), lr=LR_INIT, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60, 80], gamma=0.1)

    steps_per_epoch = len(train_dataset) // args.batch_size

    GLOBAL_STEPS = 1

    model.train()
    decoder = {k: 0 for k in range(10)}

    for epoch in range(args.epochs):
        fit_one_epoch(
            epoch,
            model,
            train_loader,
            optimizer,
            (GLOBAL_STEPS, steps_per_epoch),
        )
        torch.save(
            model.state_dict(),
            "{}/hrnet18v2_ranger{}.pth".format(args.output_path, epoch + 1),
        )
        scheduler.step()


if __name__ == "__main__":
    main()