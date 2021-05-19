import math
import numpy as np
import cv2
import os


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        """
        Average meter constructor

        """
        self.reset()

    def reset(self):
        """
        Reset AverageMeter

        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update value 

        Parameters:
            val (float): Value to add in the meter 
            n (int): Batch size

        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def tiling(img_path, output_dir, tile_size=(869, 1302), offset=(782, 1172), preprocess=False, save=False):
    """
    Function used to tile image (split image) in order to increase training samples

    Parameters:

        img_path (str): Path to image
        output_dir (str): Path to save tile images
        tile_size (tuple): size of the tile 
        offset (tuple): Offset is used to merge boundary of the previous and next tile 
        preprocess (bool): True for preprocessing and false for postprocessing 
        save (bool): save tile images or not

    Returns:
            cropped_image (np.ndarray): tiled image 


    """

    img = cv2.imread(img_path)
    data = img_path.split('/')
    if preprocess:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        path = output_dir + '/'
    else:
        path = os.path.join(output_dir, 'tiling_output')
        os.makedirs(path, exist_ok=True)
    img_shape = img.shape
    for i in range(int(math.ceil(img_shape[0]/(offset[0] * 1.0)))):
        for j in range(int(math.ceil(img_shape[1]/(offset[1] * 1.0)))):
            cropped_img = img[offset[0]*i:min(offset[0]*i+tile_size[0], img_shape[0]),
                              offset[1]*j:min(offset[1]*j+tile_size[1], img_shape[1])]

            if cropped_img.shape[0:2] != tile_size:
                if cropped_img.shape[0] != tile_size[0] and cropped_img.shape[1] != tile_size[1]:
                    cropped_img = img[img_shape[0]-tile_size[0]:img_shape[0],
                                      img_shape[1]-tile_size[1]:img_shape[1]]
                elif cropped_img.shape[0] != tile_size[0]:
                    cropped_img = img[img_shape[0]-tile_size[0]:img_shape[0],
                                      offset[1]*j:min(offset[1]*j+tile_size[1], img_shape[1])]
                elif cropped_img.shape[1] != tile_size[1]:
                    cropped_img = img[offset[0]*i:min(offset[0]*i+tile_size[0], img_shape[0]),
                                      img_shape[1]-tile_size[1]:img_shape[1]]
            if save:
                cv2.imwrite(path + data[-1][:-4] + "_" + str(i) +
                            "_" + str(j) + data[-1][-4:], cropped_img)
            else:
                return cropped_img

############################## Stiching ##############################################


def stiching(img_dir, height, width, tile_size=(869, 1302), offset=(782, 1172)):
    """
    Function used to stitch images

    Parameters:

        img_dir (str): path to the directory of tile images
        height (int): height of the output image
        width (int): width of the output image
        tile_size (tuple): size of the tile 
        offset (tuple): Offset is used to merge boundary of the previous and next tile 

    Returns:
            stitched_image (np.ndarray): return stitched image

    """

    nimg = np.ones((height, width), dtype=np.float32)
    img_shape = nimg.shape
    for ipath in sorted(os.listdir(img_dir)):
        path = os.path.join(img_dir, ipath)
        img = cv2.imread(path, 0)
        h, w = img.shape
        data = path.split('_')
        y, x = int(data[-2]), int(data[-1][:-4])

        if min(offset[0]*y+tile_size[0], img_shape[0]) - offset[0]*y < h and min(offset[1]*x+tile_size[1], img_shape[1]) - offset[1]*x < w:
            nimg[img_shape[0]-tile_size[0]:img_shape[0],
                 img_shape[1]-tile_size[1]:img_shape[1]] = img
        elif min(offset[0]*y+tile_size[0], img_shape[0]) - offset[0]*y < h:
            nimg[img_shape[0]-tile_size[0]:img_shape[0],
                 offset[1]*x:min(offset[1]*x+tile_size[1], img_shape[1])] = img
        elif min(offset[1]*x+tile_size[1], img_shape[1]) - offset[1]*x < w:
            nimg[offset[0]*y:min(offset[0]*y+tile_size[0], img_shape[0]),
                 img_shape[1]-tile_size[1]:img_shape[1]] = img
        else:
            nimg[offset[0]*y:min(offset[0]*y+tile_size[0], img_shape[0]),
                 offset[1]*x:min(offset[1]*x+tile_size[1], img_shape[1])] = img
    return nimg
