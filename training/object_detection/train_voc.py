from model.fcos import FCOSDetector
import torch
from dataloader.custom_dataset import YoloDataset
import math, time
from utils.ranger import Ranger
from torch import nn
from utils.utils import lr_func
import os
import argparse
from eval import validate_one_epoch


def fit_one_epoch(epoch, model, train_loader, optimizer, steps):
    GLOBAL_STEPS, steps_per_epoch = steps
    for epoch_step, data in enumerate(train_loader):

        batch_imgs, batch_boxes, batch_classes = data
        batch_imgs = batch_imgs.cuda()

        batch_boxes = batch_boxes.cuda()
        batch_classes = batch_classes.cuda()
        #        batch_mask = batch_mask.cuda()
        lr = lr_func()
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
        "--weight_path",
        type=str,
        required=True,
        help="path to weight pth file",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="output directory to save model",
    )

    args = parser.parse_args()
    train_dataset = YoloDataset(args.train_txt)
    val_dataset = YoloDataset(
        "/media/sanchit/datasets/Our-collected-dataset/plate_data_download/Data_public/Extras/train_yolo.txt",
        train=False,
    )

    model = FCOSDetector(mode="training").cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = Ranger(model.parameters(), lr=1e-4)

    BATCH_SIZE = 2
    EPOCHS = 200
    WARMPUP_STEPS_RATIO = 0.12

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=val_dataset.collate_fn,
    )

    steps_per_epoch = len(train_dataset) // BATCH_SIZE
    TOTAL_STEPS = steps_per_epoch * EPOCHS
    WARMPUP_STEPS = TOTAL_STEPS * WARMPUP_STEPS_RATIO

    GLOBAL_STEPS = 1
    LR_INIT = 5e-4
    LR_END = 1e-6

    model.train()

    os.makedirs(ckpt_path, exist_ok=True)

    for epoch in range(EPOCHS):
        fit_one_epoch(
            epoch, model, train_loader, optimizer, (GLOBAL_STEPS, steps_per_epoch)
        )
        torch.save(
            model.state_dict(),
            "{}/hrnet18v2_ranger{}.pth".format(ckpt_path, epoch + 1),
        )
