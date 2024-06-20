from torch import nn
from torchvision import models
import torch.nn.functional as F
from torchvision.models import ResNet34_Weights


class BB_model(nn.Module):

    def __init__(self):
        super(BB_model, self).__init__()

        # 选取resnet34前8个层：这样的分割可能是为了在不改变早期特征学习的情况下，调整模型的后期部分以适应新的任务需求。
        resnet = models.resnet34(
            weights=ResNet34_Weights.DEFAULT)  # 加载ResNet34模型，pretrained=True,模型的权重是基于ImageNet数据集预先训练好的
        layers = list(resnet.children())[:8]  # 将ResNet34模型分解为多个子模块（children），并选取前8个层。
        self.features1 = nn.Sequential(*layers[:6])  # 前6层（通常是卷积层和池化层）被包装进self.features1
        self.features2 = nn.Sequential(*layers[6:])  # 剩下的2层（通常是全连接前的最后一层卷积和全局平均池化层）构成self.features2

        # 定义了两个独立的全连接层序列（Sequential），分别用于分类和边界框预测。
        self.classifier = nn.Sequential(nn.BatchNorm1d(512),
                                        nn.Linear(512, 4))  # 批量归一化层：对输入数据的最后一个维度归一化为零均值和单位方差，线性层输出分类结果
        self.bb = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))  # ..., 线性层输出边界框四个坐标

    def forward(self, x):
        '''
        nn.AdaptiveAvgPool2d：自适应平均池化（Adaptive Average Pooling）:
        特别是在卷积神经网络（CNN）的尾部，在全连接层（Fully Connected Layer）之前，
        用于将任意大小的输入特征图（feature map）转换成固定大小的输出特征图，设置为(1, 1)意味着每个通道得到一个单一值，输出为(batch_size, channels, 1, 1)；
        x.view(x.shape[0], -1)：展平操作，(batch_size, channels, height, width) → (batch_size, channels*heigh* width)
        '''
        x = self.features1(x)
        x = self.features2(x)
        x = F.relu(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x), self.bb(x)
