import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil

def bifurcate(img):
    #img = cv2.resize(img,(94,24))
    main_img = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = img.shape[0]
    w = img.shape[1]
    if w/h>1.75:
        return (np.zeros((1,1)))
    img = img[int(0.2*h):int(0.8*h),int(0.2*w):int(0.8*w)]
    img[img>=80] = 255
    img[img<80] = 0
    # im = plt.imshow(img,cmap='gray', vmin=0, vmax=255)
    # plt.show()

    b_start_flag,f1 = False,False
    w_mid_flag,f2 = False,False
    b_end_flag,f3 = False,False
    div_index_cropped = -1
    for i in range(len(img)):
        if np.any(img[i, :] == 0) and not f1:
            b_start_flag,f1 = True,True
        if np.all(img[i, :] == 255) and b_start_flag and not f2:
            w_mid_flag,f2 = True,True
            div_index_cropped = i
        if np.any(img[i, :] == 0) and w_mid_flag and not f3:
            b_end_flag,f3 = True,True


    if b_start_flag and w_mid_flag and b_end_flag:
        div_index_original = int(div_index_cropped+int(0.22*h))
        imgs = recreate(h,w,div_index_original,main_img)
        return imgs
    return (np.zeros((1,1)))

def recreate(h,w,index,img):
    im1 = img[:index,int(0.1*w):int(0.9*w)]
    im2 = img[index:,int(0.1*w):int(0.9*w)]
    mean_h = int((im1.shape[0] + im2.shape[0]) / 2)
    if im1.shape[0] > im2.shape[0]:
        im2 = cv2.resize(im2,(w,im1.shape[0]))
    else:
        im1 = cv2.resize(im1,(w,im2.shape[0]))
    #im3 = np.concatenate((im1, im2), axis=1)   
    # im = plt.imshow(im3,cmap='gray', vmin=0, vmax=255)
    # plt.show()
    return (im1,im2)

def main():
    path = './images_0/test'
    #os.makedirs('./2line/string_matching')
    out_path = './2line/string_matching'
    for img_name in os.listdir(path):
        img = cv2.imread(os.path.join(path,img_name))
        imgs = bifurcate(img)
        if len(imgs)==1:
            continue
        cv2.imwrite(os.path.join(out_path,'0_'+img_name),imgs[0])
        cv2.imwrite(os.path.join(out_path,'1_'+img_name),imgs[1])
if __name__ == "__main__":
    main()
        
        

