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

from Data_augmentation_Train_valid_split import transformsXY, show_corner_bb
from Dataset import RoadDataset
from Loading_resizing import generate_train_df, resize_image_bb, create_bb_array, create_mask, mask_to_bb, \
    read_image
from Model import BB_model
from Training import train_epocs, update_optimizer

# 清空，直接复制到控制台
'''
%reset -f    # 相当于 Matlab 中的 clear, -f 代表 force，即强制执行
import matplotlib.pyplot as plt
plt.close("all")    # 相当于 Matlab 中的 close all, 即关闭所有图片
%clear    # 相当于 Matlab 中的 clc, 即清空命令窗口
'''

#%% Loading
# 定义图片和注解文件夹的路径
images_path = Path('./UAV-Eagle dataset/images')
anno_path = Path('./UAV-Eagle dataset/labels')

# 生成DataFrame：df_train
df_train = generate_train_df(images_path, anno_path)
print('df_train sahpe:', df_train.shape)  # (617, 8)
print(df_train.head())

#%% Resizing
train_path_resized = Path('./UAV-Eagle dataset/images_resized')  # 新的图像存放路径
new_paths, new_bbs = [], []  # 初始化新路径和新边界框列表
for index, row in df_train.iterrows():  # 遍历df_train的每一行
    new_path, new_bb = resize_image_bb(row['filename'], train_path_resized, create_bb_array(row.values),
                                       300)  # 调整图像和边界框
    new_paths.append(new_path)  # 添加新路径到列表
    new_bbs.append(new_bb)  # 添加新边界框到列表
df_train['new_path'] = new_paths  # 将新路径添加为数据框的新列
df_train['new_bb'] = new_bbs  # 将新边界框添加为数据框的新列
print('df_train shape:',df_train.shape)
print(df_train.head())

#%% 遮罩查看
im = cv2.imread(str(df_train.values[58][0]))
bb = create_bb_array(df_train.values[58])
Y = create_mask(bb, im)

plt.figure()
plt.imshow(im)
plt.figure()
plt.imshow(Y, cmap='gray')

#%% Augmentation测试
# original
im = cv2.imread(str(df_train.values[58][8]))  # df_train的第8列是new_path
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
show_corner_bb(im, df_train.values[58][9])

# after transformation
im, bb = transformsXY(str(df_train.values[58][8]), df_train.values[58][9], True)
show_corner_bb(im, bb)

#%% 分离train和valid
df_train = df_train.reset_index()  # 重置df_train（一个Pandas DataFrame）的索引
X = df_train[['new_path', 'new_bb']]
Y = df_train['class']
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)  # train_test_split函数，该函数来自Scikit-learn库，用于将数据集划分为训练集和验证集

#%% Dataset
train_ds = RoadDataset(X_train['new_path'], X_train['new_bb'], y_train, transforms=True)
valid_ds = RoadDataset(X_val['new_path'], X_val['new_bb'], y_val)

batch_size = 16
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=batch_size)

#%% Train
model = BB_model().cuda()
parameters = filter(lambda p: p.requires_grad, model.parameters())  # 筛选出模型中所有需要梯度更新的参数
optimizer = torch.optim.Adam(parameters, lr=0.008)
train_epocs(model, optimizer, train_dl, valid_dl, epochs=10)
update_optimizer(optimizer, 0.001)
train_epocs(model, optimizer, train_dl, valid_dl, epochs=5)

#%% Test测试集
# 选择并resize要测试的图像
im = read_image('./UAV-Eagle dataset/images_resized/gopro_scene_0365.png')
im = cv2.resize(im, (int(1.49*300), 300))
cv2.imwrite('./UAV-Eagle dataset/UAV-Eagle_test/gopro_scene_0365.png', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

# 将要测试的图象构建为一个RoadDataset数据集
# 仅包含一个样本，图像路径指向刚刚保存的图片，边界框(bb)初始化为全零数组，表示没有确切的边界框信息（或需要模型预测），类别标签(y)设为0。
test_ds = RoadDataset(pd.DataFrame([{'path':'./UAV-Eagle dataset/images_resized/gopro_scene_0365.png'}])['path'],pd.DataFrame([{'bb':np.array([0,11,20,15])}])['bb'],pd.DataFrame([{'y':[0]}])['y'])
x, y_class, y_bb = test_ds[0]  # 从test_ds中取出第一个也是唯一一个样本，解包得到图像数据x、类别标签y_class和边界框坐标y_bb。
xx = torch.FloatTensor(x[None,])  # 将图像数据x转换为一个形状为(1, C, W, H)的PyTorch浮点张量，以便通过模型进行预测。
print(xx.shape)

# prediction
model = BB_model().cuda()
model.eval()
out_class, out_bb = model(xx.cuda())

# predicted class
print(torch.max(out_class, 1))

# predicted bounding box
bb_hat = out_bb.detach().cpu().numpy()
bb_hat = bb_hat.astype(int)
show_corner_bb(im, bb_hat[0])