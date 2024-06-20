import torch
import torch.nn.functional as F


def update_optimizer(optimizer, lr):
    for i, param_group in enumerate(optimizer.param_groups):  # enumerate函数用于遍历优化器中的每个参数组，并为每个参数组提供一个索引i和对应的参数组字典param_group
        param_group["lr"] = lr  # param_group["lr"] = lr这一行代码修改了当前遍历到的参数组字典中的学习率项，将其设置为传入的lr值。


def val_metrics(model, valid_dl, C=4):
    '''
    model: 要评估的神经网络模型，它应该已经训练好了并且包含了分类和边界框预测的功能。
    valid_dl: 验证数据加载器（DataLoader），提供批处理的验证数据集，包含图像、类别标签和边界框坐标。
    C: 一个可选参数，默认值为1000，通常对应于分类任务中的类别数量。在损失计算中用于调整分类损失和边界框回归损失的平衡。
    '''
    model.eval()
    total = 0       # 总样本数
    sum_loss = 0    # 总损失
    correct = 0     # 正确分类的样本数
    for x, y_class, y_bb in valid_dl:   # 遍历验证集各批次
        batch = y_class.shape[0]
        x = x.cuda().float()
        y_class = y_class.cuda()
        y_bb = y_bb.cuda().float()
        out_class, out_bb = model(x)  # 前向传播

        loss_class = F.cross_entropy(out_class, y_class, reduction="sum")   # 分类损失使用交叉熵损失函数F.cross_entropy计算
        loss_bb = F.l1_loss(out_bb, y_bb, reduction="none").sum(1)          # 边界框回归损失使用L1损失函数F.l1_loss计算，边界框有4个坐标还要.sum(1)
        loss_bb = loss_bb.sum()                                             # batch内所有样本的损失求和
        loss = loss_class + loss_bb/C                                       # 将分类损失除以类别数目C后，和边界框回归损失相加得到总损失
        _, pred = torch.max(out_class, 1)                                   # 使用torch.max找出预测类别，并与真实类别比较，累加正确预测的数量

        correct += pred.eq(y_class).sum().item()
        sum_loss += loss.item()
        total += batch

    return sum_loss/total, correct/total                                    # 返回：平均损失（sum_loss/total）和平均准确率（correct/total）


def train_epocs(model, optimizer, train_dl, valid_dl, epochs=10, C=4):
    '''
    model: 需要训练的深度学习模型，它应包含对图像分类和边界框预测的功能。
    optimizer: 用于更新模型参数的优化器，如Adam、SGD等。
    train_dl: 训练数据加载器，提供批量的训练数据。
    valid_dl: 验证数据加载器，用于评估模型在未见过数据上的表现。
    epochs=10: 训练的迭代周期数，默认为10次。
    C=4: 分类任务中的类别总数，用于损失计算时的归一化。
    '''

    idx = 0  # 总批次数
    for i in range(epochs):

        model.train()
        total = 0  # 总样本数
        sum_loss = 0

        for x, y_class, y_bb in train_dl:       # 遍历训练数据加载器中的每一个批次数据 (x输入图像, y_class类别标签, y_bb边界框)

            batch = y_class.shape[0]
            x = x.cuda().float()
            y_class = y_class.cuda()
            y_bb = y_bb.cuda().float()
            out_class, out_bb = model(x)        # 前向传播

            loss_class = F.cross_entropy(out_class, y_class, reduction="sum")   # 分类损失 loss_class 使用交叉熵损失函数 F.cross_entropy 计算，且损失求和（reduction="sum"）。
            loss_bb = F.l1_loss(out_bb, y_bb, reduction="none").sum(1)          # 边界框回归损失 loss_bb 使用 L1 损失函数 F.l1_loss 计算，对每个样本的损失先进行求和（按行 sum(1)）
            loss_bb = loss_bb.sum()                                             # 再整体求和
            loss = loss_class + loss_bb/C                                       # 最后除以类别数 C 以平衡分类和回归损失

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            idx += 1
            total += batch
            sum_loss += loss.item()

        train_loss = sum_loss/total
        val_loss, val_acc = val_metrics(model, valid_dl, C)
        print("train_loss %.3f val_loss %.3f val_acc %.3f" % (train_loss, val_loss, val_acc))

    return sum_loss/total


