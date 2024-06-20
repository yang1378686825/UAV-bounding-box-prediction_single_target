import numpy as np
from torch.utils.data import Dataset

from Data_augmentation_Train_valid_split import transformsXY


def normalize(im):
    """
    对输入的图像数据进行归一化处理，其依据是ImageNet数据集的统计信息:imagenet_stats
    对于图像数组中的每个像素值，首先从原始值中减去ImageNet数据集中对应通道的平均值（去均值），然后除以对应通道的标准差（归一化）。
    使得处理后的图像数据具有零均值和单位方差,尤其是在使用预训练的基于ImageNet的模型时,可以最大化模型的性能。
    """
    imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
    return (im - imagenet_stats[0])/imagenet_stats[1]

class RoadDataset(Dataset):
    '''
    RoadDataset继承了PyTorch的Dataset基类，
    这意味着它需要实现一些基本方法，如__init__, __len__, 和 __getitem__，以便能够与PyTorch的数据加载器（DataLoader）无缝集成，用于训练和验证模型。
    paths: 包含图像路径的数组或列表。
    bb: 与图像对应的边界框信息，通常是一个二维数组，每行代表一个图像的边界框坐标。
    y: 数据的类别标签，与paths一一对应。
    transforms: 布尔值，指明是否在加载数据时应用数据增强变换。
    '''
    def __init__(self, paths, bb, y, transforms=False):
        self.transforms = transforms
        self.paths = paths.values
        self.bb = bb.values
        self.y = y.values

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        '''
        根据给定的索引获取单个数据样本
        idx: 索引值，当数据加载器请求数据时提供。
        return x, y_class, y_bb：x是预处理后的图像，y_class是图像的类别标签，y_bb是图像的边界框坐标
        '''
        path = self.paths[idx]
        y_class = self.y[idx]
        x, y_bb = transformsXY(path, self.bb[idx], self.transforms)  # 旋转、翻转、裁剪变换，并返回变换后的图像x和对应的边界框y_bb
        x = normalize(x)  # 对图像x归一化
        x = np.rollaxis(x, 2)  # 将图像的通道维度从最后一个位置移到第一个位置，即形状从(H, W, C)转换为(C, H, W)。
        return x, y_class, y_bb

