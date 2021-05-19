"""
@Author: xxxmy
@Github: github.com/VectXmy
@Date: 2019-09-26
@Email: xxxmy@foxmail.com
"""
from model.fcos import FCOSDetector
import torch


from dataloader.custom_dataset import YoloDataset

import math, time
from torch.utils.tensorboard import SummaryWriter
from ranger import Ranger

train_dataset = YoloDataset(
    "/media/sanchit/datasets/Our-collected-dataset/plate_data_download/Data_public/Extras/train_yolo.txt"
)
val_dataset = YoloDataset(
    "/media/sanchit/datasets/Our-collected-dataset/plate_data_download/Data_public/Extras/train_yolo.txt",
    train=False,
)

model = FCOSDetector(mode="training").cuda()

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
    val_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=val_dataset.collate_fn
)
steps_per_epoch = len(train_dataset) // BATCH_SIZE
TOTAL_STEPS = steps_per_epoch * EPOCHS
WARMPUP_STEPS = TOTAL_STEPS * WARMPUP_STEPS_RATIO

GLOBAL_STEPS = 1
LR_INIT = 5e-4
LR_END = 1e-6

writer = SummaryWriter(log_dir="./logs")


def lr_func():
    if GLOBAL_STEPS < WARMPUP_STEPS:
        lr = GLOBAL_STEPS / WARMPUP_STEPS * LR_INIT
    else:
        lr = LR_END + 0.5 * (LR_INIT - LR_END) * (
            (
                1
                + math.cos(
                    (GLOBAL_STEPS - WARMPUP_STEPS)
                    / (TOTAL_STEPS - WARMPUP_STEPS)
                    * math.pi
                )
            )
        )
    return float(lr)


model.train()
scaler = torch.cuda.amp.GradScaler()
amp = False
ckpt_path = "./pothole_collected_data_batch1"
import os

os.makedirs(ckpt_path, exist_ok=True)
for epoch in range(EPOCHS):
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
        if amp:
            with torch.cuda.amp.autocast():
                losses = model([batch_imgs, batch_boxes, batch_classes])
            loss = losses[-1]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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

    torch.save(
        model.state_dict(),
        "{}/hrnet18v2_ranger{}.pth".format(ckpt_path, epoch + 1),
    )
