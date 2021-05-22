import os
import shutil
import pandas as pd
import numpy as np
import cv2 as cv
from PIL import Image
import argparse
import re
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(description="parameters for dataset preprocessing")
    parser.add_argument(
        "--input_dir", default="", help="Input path (contains imgfolder and label csv)"
    )
    parser.add_argument(
        "--resize", default=False, type=bool, help="resize images to 94,24"
    )
    parser.add_argument(
        "--output_dir",
        default="./images/",
        help="a folder containing train and test folders will be created here",
    )  # don't pass for easy training
    parser.add_argument(
        "--verbose", default=False, type=bool, help="Set to true for verbose"
    )
    parser.add_argument("--test_size", default=0.15, type=bool, help="test size ratio")

    args = parser.parse_args()
    return args


def label_check(label):
    if len(label) < 4:
        return 0
    return 1

    if label[0:2] == "DL":
        delhi_pt = "\d{4,4}$"
        dlval = re.search(delhi_pt, label)
        if dlval is None or len(dlval.group()) < 8:
            return 0
        else:
            return 1
    pattern = "(([A-Za-z]){2,3}(|-)(?:[0-9]){1,2}(|-)(?:[A-Za-z]){2}(|-)([0-9]){1,4})|(([A-Za-z]){2,3}(|-)([0-9]){1,4})"
    val = re.search(pattern, label)
    if val is None or len(val.group()) < 8:
        return 0
    else:
        return 1


def size_check(ipath):
    return 1
    img = cv.imread(ipath)
    height = img.shape[0]
    width = img.shape[1]
    # print(height,width)
    if width < 65 and height < 15:
        return 0
    return 1


def preprocess():
    args = get_parser()
    idr = os.path.expanduser(args.input_dir)
    odr = os.path.expanduser(args.output_dir)
    if not os.path.exists(odr):
        os.mkdir(odr)
    imgfolder, dfpath = None, None
    for item in os.listdir(idr):
        if item[-2:] == "py":
            continue
        if os.path.isdir(os.path.join(idr, item)):
            imgfolder = os.path.join(idr, item)
        else:
            dfpath = os.path.join(idr, item)
    if dfpath[-3:] != "csv":
        df = pd.read_excel(dfpath)
    else:
        df = pd.read_csv(dfpath, encoding="unicode_escape")
    df = df.astype(str)
    allFileNames = os.listdir(imgfolder)
    np.random.shuffle(allFileNames)
    train_FileNames, test_FileNames = np.split(
        np.array(allFileNames), [int(len(allFileNames) * (1 - args.test_size))]
    )
    print("Total images: ", len(allFileNames))
    print("Training: ", len(train_FileNames))
    print("Testing: ", len(test_FileNames))
    train_FileNames = [imgfolder + "/" + name for name in train_FileNames.tolist()]
    test_FileNames = [imgfolder + "/" + name for name in test_FileNames.tolist()]
    os.makedirs(odr + "/train")
    os.makedirs(odr + "/test")
    print("Copying train:")
    for name in tqdm(train_FileNames):
        shutil.copy(name, odr + "/train")
    print("Copying test:")
    for name in tqdm(test_FileNames):
        shutil.copy(name, odr + "/test")
    count = 0
    # for dirs in os.listdir(odr):
    #     print(f"Preprocessing {dirs}:")
    #     for img in tqdm(os.listdir(os.path.join(odr + dirs))):
    #         ipath = os.path.join(odr, dirs, img)
    #         # img,_ = os.path.splitext(img) #uncomment this line if csv/xls has imgname without extension(eg 1 instead of 1.png)
    #         _, ext = os.path.splitext(ipath)
    #         try:
    #             label = df[df.iloc[:, 0] == img].iloc[0, 1]
    #         except:
    #             count += 1
    #             os.remove(ipath)
    #             continue
    #         label = label.replace("/0", "")
    #         label = "".join(e for e in label if e.isalnum())
    #         if (
    #             img not in df.iloc[:, 0].tolist()
    #             or label_check(label) == 0
    #             or size_check(ipath) == 0
    #         ):
    #             count += 1
    #             if args.verbose:
    #                 print(
    #                     f"Image not found/ Image too small/ Label error: Discarding image:{img}"
    #                 )
    #             try:
    #                 os.remove(ipath)
    #             except:
    #                 print(ipath)
    #             continue

    #         if args.resize:
    #             im = Image.open(ipath)
    #             imResize = im.resize((94, 24), Image.ANTIALIAS)

    #         tpath = os.path.join(odr, dirs, label + ext)
    #         if os.path.exists(tpath):
    #             tpath = os.path.join(odr, dirs, label + "_" + ext)
    #             if os.path.exists(tpath):
    #                 os.remove(ipath)
    #                 count += 1
    #                 continue
    #             os.rename(ipath, tpath)
    #             continue
    #             if os.path.getsize(tpath) > os.path.getsize(ipath):
    #                 os.remove(ipath)
    #             else:
    #                 if args.resize:
    #                     os.remove(tpath)
    #                     os.remove(ipath)
    #                     if tpath[:-3] == "png":
    #                         imResize.save(ipath, "PNG", quality=100)
    #                     else:
    #                         imResize.save(ipath, "JPEG", quality=100)
    #                 else:
    #                     os.remove(tpath)
    #                     os.rename(ipath, tpath)
    #             count += 1
    #             if args.verbose:
    #                 print(f"Discarding duplicate image:{img}")
    #             continue

    #         if args.resize:
    #             os.remove(ipath)
    #             if tpath[:-3] == "png":
    #                 imResize.save(tpath, "PNG", quality=100)
    #             else:
    #                 imResize.save(tpath, "JPEG", quality=100)
    #             continue

    #         os.rename(ipath, tpath)
    # print(f"Discarded {count} images in total.")


if __name__ == "__main__":
    preprocess()
