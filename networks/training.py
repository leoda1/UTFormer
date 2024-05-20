
# -*- coding: UTF-8 -*-
"""
===================================================================================
@author : Leoda
@Date   : 2024/01/15 21:23:15
@Project -> : learn$
==================================================================================
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

def CE_Loss(inputs, target, cls_weights, num_classes=21):
    """
    计算交叉熵损失
    首先，它确保输入和目标的空间维度（即高和宽）匹配，必要时会调整输入的大小。然后，它将输入和目标重新调整为适合
    交叉熵损失函数的形状，并使用PyTorch的nn.CrossEntropyLoss来计算损失。这个损失函数通常用于训练分类模型，
    其中cls_weights可以用于给不同类别分配不同的权重，从而处理类别不平衡问题。
    """
    # 获取输入的大小，其中n是批次大小，c是通道数，h和w分别是输入的高和宽
    n, c, h, w = inputs.size()
    # 获取目标的大小，其中nt是批次大小，ht和wt分别是目标的高和宽
    nt, ht, wt = target.size()

    # 如果输入和目标的高和宽不匹配，则通过双线性插值调整输入的大小以匹配目标
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    # 转置输入，并调整其形状以便后续计算，变成一个二维张量，其中每一行对应一个位置的类别分数
    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    # 调整目标的形状，变成一维张量
    temp_target = target.view(-1)

    # 使用PyTorch的交叉熵损失函数，其中cls_weights是每个类别的权重，ignore_index=num_classes表示忽略等于num_classes的目标标签
    CE_loss = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)

    # 返回计算得到的交叉熵损失
    return CE_loss

def Focal_Loss(inputs, target, cls_weights, num_classes=21, alpha=0.5, gamma=2):
    """
    计算类别不平衡问题的损失函数
    通过引入一个调节因子，该因子会减少容易分类样本的权重，并增加错误分类样本的权重。这种调整使得模型在训练时更关
    注难以分类的样本。参数alpha和gamma是用于调节这种权重的超参数，可用于控制对不平衡类别的关注程度。
    """
    # 获取输入的大小，其中n是批次大小，c是通道数，h和w分别是输入的高和宽
    n, c, h, w = inputs.size()
    # 获取目标的大小，其中nt是批次大小，ht和wt分别是目标的高和宽
    nt, ht, wt = target.size()

    # 如果输入和目标的高和宽不匹配，则通过双线性插值调整输入的大小以匹配目标
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    # 转置输入，并调整其形状以便后续计算，变成一个二维张量，其中每一行对应一个位置的类别分数
    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    # 调整目标的形状，变成一维张量
    temp_target = target.view(-1)

    # 计算交叉熵损失的负对数概率，用reduction='none'来保持结果的形状
    logpt = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes, reduction='none')(temp_inputs, temp_target)
    # 计算每个类别的概率
    pt = torch.exp(logpt)

    # 如果提供了alpha参数，则将其乘到负对数概率上
    if alpha is not None:
        logpt *= alpha

    # 计算Focal Loss，通过增强错误分类样本的权重来解决类别不平衡问题
    loss = -((1 - pt) ** gamma) * logpt
    # 计算损失的均值
    loss = loss.mean()

    # 返回计算得到的Focal Loss
    return loss

def Dice_loss(inputs, target, beta=1, smooth=1e-5):
    """
    计算基于Dice系数,用于衡量两个样本的相似度
    """
    # 获取输入的大小，其中n是批次大小，c是通道数，h和w分别是输入的高和宽
    n, c, h, w = inputs.size()
    # 获取目标的大小，其中nt是批次大小，ht和wt分别是目标的高和宽，ct是目标的通道数
    nt, ht, wt, ct = target.size()

    # 如果输入和目标的高和宽不匹配，则通过双线性插值调整输入的大小以匹配目标
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    # 使用softmax激活函数处理输入，并调整其形状
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    # 调整目标的形状
    temp_target = target.view(n, -1, ct)

    # --------------------------------------------#
    #   计算dice loss
    # --------------------------------------------#
    # 计算真正例（true positive）
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    # 计算假正例（false positive）
    fp = torch.sum(temp_inputs                        , axis=[0, 1]) - tp
    # 计算假负例（false negative）
    fn = torch.sum(temp_target[..., :-1]              , axis=[0, 1]) - tp

    # 计算Dice系数，beta是调节参数，smooth是平滑参数防止分母为零
    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    # 计算Dice损失
    dice_loss = 1 - torch.mean(score)

    # 返回Dice损失
    return dice_loss

def weights_init(net, init_type='normal', init_gain=0.02):
    """
    根据指定的类型对神经网络进行权重初始化
    不同的初始化方法可能对训练过程的收敛速度和最终性能有重要影响。支持的初始化方法包括正态分布初始化（normal）
    、Xavier初始化（xavier）、Kaiming初始化（kaiming）和正交初始化（orthogonal）。对于BatchNorm2d
    层，权重用正态分布初始化，偏置用常数初始化
    """
    # 定义初始化函数init_func
    def init_func(m):
        # 获取模块的类名
        classname = m.__class__.__name__
        # 如果模块具有权重属性，并且是卷积层
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            # 根据初始化类型选择不同的初始化方法
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                # 如果初始化类型未定义，则引发错误
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        # 如果模块是BatchNorm2d层
        elif classname.find('BatchNorm2d') != -1:
            # 对权重进行正态分布初始化，均值为1.0，标准差为0.02
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            # 对偏置进行常数初始化，值为0.0
            torch.nn.init.constant_(m.bias.data, 0.0)

    # 打印初始化类型
    print('initialize network with %s type' % init_type)
    # 对网络中的所有模块应用初始化函数init_func
    net.apply(init_func)

#定义学习率调度器
def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.1, warmup_lr_ratio=0.1, no_aug_iter_ratio=0.3, step_num=10):
    """
    get_lr_scheduler函数接受学习率衰减类型、初始学习率、最小学习率、总迭代次数以及其他可选参数，并返回一个用于调度学习率的函数
    用于cosine衰减策略或阶梯衰减策略，根据训练过程中的迭代次数调整学习率。这有助于在训练过程中逐渐降低学习率，从而促进模型的收敛
    """
    # yolox_warm_cos_lr函数实现了warmup和cosine衰减策略
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        # 如果在warmup阶段（前warmup_total_iters次迭代），则使用二次函数逐渐增加学习率
        if iters <= warmup_total_iters:
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        # 如果在no_aug阶段（后no_aug_iter次迭代），则学习率设置为最小值
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        # 在其余的迭代中，学习率根据余弦衰减
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    # step_lr函数实现了阶梯式衰减
    def step_lr(lr, decay_rate, step_size, iters):
        # 验证步长是否大于1
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        # 计算衰减的次数
        n = iters // step_size
        # 应用衰减
        out_lr = lr * decay_rate ** n
        return out_lr

    # 根据所选择的学习率衰减类型，确定使用哪个衰减函数
    if lr_decay_type == "cos":
        # warmup阶段的迭代次数和起始学习率
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        # no_aug阶段的迭代次数
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        # 使用functools.partial应用yolox_warm_cos_lr
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        # 计算每一步的衰减率
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        # 计算每一步的大小
        step_size = total_iters / step_num
        # 使用functools.partial应用step_lr
        func = partial(step_lr, lr, decay_rate, step_size)

    return func  # 返回学习率调度函数

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    """
    更新优化器中的学习率
    给定优化器、学习率调度函数和当前时期，该函数将调用学习率调度函数来计算新的学习率，并将其应用于优化器的所有参数组。
    这是训练深度学习模型过程中常见的一步，通过适当调整学习率可以有助于模型更快地收敛或避免陷入局部最小值
    """
    # 通过学习率调度函数计算给定时期的学习率
    lr = lr_scheduler_func(epoch)
    # 遍历优化器的所有参数组
    for param_group in optimizer.param_groups:
        # 将每个参数组的学习率更新为计算出的新学习率
        param_group['lr'] = lr