import os
import random
import math
from datetime import datetime
from collections import Counter
import pandas as pd
import numpy as np

import cv2
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from Loading_resizing import create_mask, mask_to_bb, generate_train_df, resize_image_bb, create_bb_array


def crop(im, r, c, target_r, target_c):
    '''
    从给定的大图像im中裁剪出一个指定大小的小图像块
    区域的左上角坐标为(r, c)，尺寸为target_r×target_c（高度×宽度）
    '''
    return im[r:r+target_r, c:c+target_c]


def random_crop(x, r_pix=8):
    """
    从输入图像x中随机位置，裁剪出一个接近原尺寸但略小的图像块
    裁剪后图像的预期高度r-2*r_pix和宽度c-2*c_pix
    """
    r, c,*_ = x.shape
    c_pix = round(r_pix*c/r)
    rand_r = random.uniform(0, 1)
    rand_c = random.uniform(0, 1)
    start_r = np.floor(2*rand_r*r_pix).astype(int)
    start_c = np.floor(2*rand_c*c_pix).astype(int)
    return crop(x, start_r, start_c, r-2*r_pix, c-2*c_pix)


def center_crop(x, r_pix=8):
    '''
    从输入图像x中心位置,裁剪出一个接近原尺寸但略小的图像块
    裁剪后图像的预期高度r-2*r_pix和宽度c-2*c_pix
    '''
    r, c,*_ = x.shape
    c_pix = round(r_pix*c/r)
    return crop(x, r_pix, c_pix, r-2*r_pix, c-2*c_pix)


def rotate_cv(im, deg, y=False, mode=cv2.BORDER_REFLECT, interpolation=cv2.INTER_AREA):
    """
    使用OpenCV库对输入图像进行旋转操作
    im: 待旋转的图像，通常是一个OpenCV格式的图像（NumPy数组）。
    deg: 旋转角度，单位为度。正值表示逆时针旋转，负值表示顺时针旋转。
    """
    r,c,*_ = im.shape
    M = cv2.getRotationMatrix2D((c/2,r/2),deg,1)  # (c/2,r/2)是旋转中心（图像中心），deg是旋转角度，1是缩放因子（保持原大小）
    if y:
        return cv2.warpAffine(im, M,(c,r), borderMode=cv2.BORDER_CONSTANT)  # 使用BORDER_CONSTANT模式进行边缘处理，即将边缘填充为常数值
    return cv2.warpAffine(im,M,(c,r), borderMode=mode, flags=cv2.WARP_FILL_OUTLIERS+interpolation)  # 使用用户指定的边界处理模式mode和插值方法interpolation进行旋转，同时开启cv2.WARP_FILL_OUTLIERS标志来处理旋转后可能落在图像外的像素


def random_cropXY(x, Y, r_pix=8):
    """
    对图像x和遮罩Y一起random_crop
    """
    r, c,*_ = x.shape
    c_pix = round(r_pix*c/r)
    rand_r = random.uniform(0, 1)
    rand_c = random.uniform(0, 1)
    start_r = np.floor(2*rand_r*r_pix).astype(int)
    start_c = np.floor(2*rand_c*c_pix).astype(int)
    xx = crop(x, start_r, start_c, r-2*r_pix, c-2*c_pix)
    YY = crop(Y, start_r, start_c, r-2*r_pix, c-2*c_pix)
    return xx, YY


def transformsXY(path, bb, transforms):
    '''
    对图像数据及相应的边界框（bounding box, bb）进行一系列预处理和变换
    path: 图像文件的路径。
    bb: 边界框信息， [xmin, ymin, xmax, ymax]。
    transforms: 布尔值，指示是否应用数据增强变换。如果为True，则执行旋转、翻转和随机裁剪；如果为False，则只进行一个中心裁剪。
    '''
    x = cv2.imread(str(path)).astype(np.float32)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)/255  # 从BGR色彩空间转换为RGB色彩空间，除以255进行归一化，像素值范围变为0到1。
    Y = create_mask(bb, x)  # 创建遮罩Y
    if transforms:
        rdeg = (np.random.random()-.50)*20  # 生成一个介于-10度到10度之间的随机旋转角度rdeg
        x = rotate_cv(x, rdeg)  # 旋转
        Y = rotate_cv(Y, rdeg, y=True)
        if np.random.random() > 0.5:  # 0.5概率水平翻转
            x = np.fliplr(x).copy()
            Y = np.fliplr(Y).copy()
        x, Y = random_cropXY(x, Y)  # 随机裁剪
    else:
        x, Y = center_crop(x), center_crop(Y)
    return x, mask_to_bb(Y)  # 将处理后的遮罩Y转换回边界框格式，返回变换后的图像x和新的边界框


def create_corner_rect(bb, color='red'):
    '''
    根据边界框bb创建一个matplotlib图形中的矩形
    返回的Rectangle对象，可以直接添加到matplotlib的坐标轴Axes实例中
    '''
    bb = np.array(bb, dtype=np.float32)
    return plt.Rectangle((bb[1], bb[0]), bb[3]-bb[1], bb[2]-bb[0], color=color,
                         fill=False, lw=3)


def show_corner_bb(im, bb):
    '''
    显示一张图像（im）并在这张图像上绘制边界框bb
    '''
    plt.figure()
    plt.imshow(im)
    plt.gca().add_patch(create_corner_rect(bb))  # plt.gca() 获取当前的图形的坐标轴 (Axes 实例)，create_corner_rect(bb)函数，根据提供的边界框坐标bb创建一个矩形 patch


# test
if __name__ == '__main__':

    anno_path = Path('./road_signs/annotations')
    df_train = generate_train_df(anno_path)
    class_dict = {'speedlimit': 0, 'stop': 1, 'crosswalk': 2, 'trafficlight': 3}
    df_train['class'] = df_train['class'].apply(lambda x: class_dict[x])
    train_path_resized = Path('./road_signs/images_resized')  # 新的图像存放路径
    new_paths, new_bbs = [], []  # 初始化新路径和新边界框列表
    for index, row in df_train.iterrows():  # 遍历数据框的每一行
        new_path, new_bb = resize_image_bb(row['filename'], train_path_resized, create_bb_array(row.values), 300)  # 调整图像和边界框
        new_paths.append(new_path)  # 添加新路径到列表
        new_bbs.append(new_bb)  # 添加新边界框到列表
    df_train['new_path'] = new_paths  # 将新路径添加为数据框的新列
    df_train['new_bb'] = new_bbs  # 将新边界框添加为数据框的新列

    #original
    im = cv2.imread(str(df_train.values[68][8]))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    show_corner_bb(im, df_train.values[68][9])

    # after transformation
    im, bb = transformsXY(str(df_train.values[68][8]),df_train.values[68][9],True )
    show_corner_bb(im, bb)

    # 分离train和valid
    df_train = df_train.reset_index()  # 重置df_train（一个Pandas DataFrame）的索引
    X = df_train[['new_path', 'new_bb']]
    Y = df_train['class']
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)  # train_test_split函数，该函数来自Scikit-learn库，用于将数据集划分为训练集和验证集
    print(X_train.shape)
    print(X_train.head())