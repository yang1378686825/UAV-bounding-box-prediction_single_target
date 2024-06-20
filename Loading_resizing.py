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


def filelist(root, file_type):
    '''
    在一个给定的文件夹（root）及其所有子文件夹中查找并返回所有指定文件类型（file_type）的文件路径列表。
    '''
    return [os.path.join(directory_path, f)
            for directory_path, directory_name, files in os.walk(root)
            for f in files if f.endswith(file_type)]


def generate_train_df(images_path, anno_path):
    """
    从YOLO格式的注解文件夹中读取.txt注解文件，提取信息并生成训练数据DataFrame。
    """
    # 获取所有.txt注解文件的路径
    annotations = filelist(anno_path, '.txt')

    # 初始化一个空列表，用于存储每个注解文件的信息字典
    anno_list = []

    # 遍历所有注解文件
    for anno_path in annotations:
        # 构建对应的图片文件路径
        image_file = os.path.splitext(anno_path)[0] + ".png"
        image_full_path = os.path.join(images_path, os.path.basename(image_file))

        # 读取图片尺寸
        with Image.open(image_full_path) as img:
            width, height = img.size

        # 读取并解析YOLO格式的注解文件
        with open(anno_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                data = line.strip().split()

                # 创建一个字典来存储当前注解的信息
                anno = {}

                # 提取并保存相关信息到字典中
                anno['filename'] = image_full_path
                anno['width'] = width
                anno['height'] = height
                anno['class'] = int(data[0])
                bb_x = float(data[1])  # bounding box中心点x坐标（归一化）
                bb_y = float(data[2])  # bounding box中心点y坐标（归一化）
                bb_width = float(data[3])  # bounding box宽度（归一化）
                bb_height = float(data[4])  # bounding box高度（归一化）

                # 转换为中心点坐标和宽高的标注为左上角和右下角坐标
                anno['xmin'] = (bb_x - bb_width / 2) * width
                anno['ymin'] = (bb_y - bb_height / 2) * height
                anno['xmax'] = (bb_x + bb_width / 2) * width
                anno['ymax'] = (bb_y + bb_height / 2) * height

                # 将当前注解信息字典添加到列表中
                anno_list.append(anno)

    # 将anno_list转换为pandas DataFrame以便后续处理
    return pd.DataFrame(anno_list)


# 读取图像并转换颜色空间为RGB
def read_image(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

# 从train_df 的某一行（即某张图片），生成边界框数组
def create_bb_array(x):
    return np.array([x[5], x[4], x[7], x[6]])  # 提取xmin, ymin, xmax, ymax   # ?ymin, xmin, ymax, xmax

# 根据边界框创建遮罩图
def create_mask(bb, x):
    """
    根据边界框创建与图像尺寸相同的遮罩
    bb: 一个包含边界框信息的NumPy数组，通常形式为 [y_min, x_min, y_max, x_max]，分别代表边界框左上角和右下角的坐标。
    x: 原始图像的数组表示，通常是从OpenCV读取的图像，我们关注的是其形状信息来创建相同尺寸的遮罩。
    """
    rows, cols, *_ = x.shape  # 获取图像的行数、列数
    Y = np.zeros((rows, cols))  # 初始化全零数组作为遮罩,其中0表示背景，后续步骤会将边界框内的部分设为1。
    bb = bb.astype(np.int32)  # 确保边界框坐标为整数
    Y[bb[0]:bb[2], bb[1]:bb[3]] = 1.  # 在边界框范围内设置遮罩值为1
    return Y

# 将遮罩转换回边界框坐标
def mask_to_bb(Y):
    """
    将遮罩Y转换回边界框坐标，假设背景为0，对象为非零值
    """
    cols, rows = np.nonzero(Y)  # 获取非零元素的列和行索引
    if len(cols) == 0:  # 如果没有找到非零元素（即无对象）
        return np.zeros(4, dtype=np.float32)  # 返回4个零，代表零边界框
    top_row, left_col = np.min(rows), np.min(cols)  # 找到左上角坐标
    bottom_row, right_col = np.max(rows), np.max(cols)  # 找到右下角坐标
    return np.array([left_col, top_row, right_col, bottom_row], dtype=np.float32)


# 调整图像大小，并相应调整边界框，然后写入新路径
def resize_image_bb(read_path, write_path, bb, sz):
    """
    read_path: 原始图像的文件路径。
    write_path: 调整大小后的图像将被保存的目录路径。
    bb: 原始图像中目标对象的边界框坐标，格式为 [y_min, x_min, y_max, x_max]。
    sz: 指定调整后图像的高度，宽度会为1.49倍sz。
    """
    read_path = Path(read_path)  # 确保read_path是Path对象
    write_path = Path(write_path)  # 确保write_path也是Path对象
    im = read_image(read_path)  # 读取图像
    im_resized = cv2.resize(im, (int(1.49*sz), sz))  # 按比例调整图像大小
    Y_resized = cv2.resize(create_mask(bb, im), (int(1.49*sz), sz))  # 创建遮罩，然后调整遮罩大小
    new_path = str(write_path / read_path.parts[-1])  # 构建新路径：write_path+原文件名
    cv2.imwrite(new_path, cv2.cvtColor(im_resized, cv2.COLOR_RGB2BGR))  # 保存调整后的图像（之前运行已经保存了因而注释掉本行和上方im_resized = ...）
    return new_path, mask_to_bb(Y_resized)  # 返回新路径和新边界框






