import math
import torch
import torch.nn as nn
from copy import deepcopy


class Conv(nn.Module):
    """卷积-批归一化-LeakyReLU激活函数组合模块
    
    Args:
        in_ch (int): 输入通道数
        out_ch (int): 输出通道数
        k (int): 卷积核大小
        p (int): 填充大小
        s (int): 步长
        d (int): 膨胀系数
        g (int): 分组卷积的组数
        act (bool): 是否使用激活函数
    """
    def __init__(self, in_ch, out_ch, k=1, p=0, s=1, d=1, g=1, act=True):
        super(Conv, self).__init__()
        if act:
            self.convs = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, dilation=d, groups=g),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.1, inplace=True)
            )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, dilation=d, groups=g),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        return self.convs(x)


class UpSample(nn.Module):
    """上采样模块
    
    Args:
        size (tuple): 输出尺寸
        scale_factor (float): 放大倍数
        mode (str): 插值模式
        align_corner (bool): 是否对齐角点
    """
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corner=None):
        super(UpSample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corner = align_corner

    def forward(self, x):
        return torch.nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, 
                                                mode=self.mode, align_corners=self.align_corner)


class reorg_layer(nn.Module):
    """重组层,用于将特征图在通道维度重组
    
    Args:
        stride (int): 重组步长
    """
    def __init__(self, stride):
        super(reorg_layer, self).__init__()
        self.stride = stride

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        _height, _width = height // self.stride, width // self.stride
        
        # 重组特征图的维度
        x = x.view(batch_size, channels, _height, self.stride, _width, self.stride).transpose(3, 4).contiguous()
        x = x.view(batch_size, channels, _height * _width, self.stride * self.stride).transpose(2, 3).contiguous()
        x = x.view(batch_size, channels, self.stride * self.stride, _height, _width).transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, _height, _width)

        return x


class SPP(nn.Module):
    """空间金字塔池化模块
    
    使用不同大小的最大池化窗口,实现多尺度特征提取
    """
    def __init__(self):
        super(SPP, self).__init__()

    def forward(self, x):
        x_1 = torch.nn.functional.max_pool2d(x, 5, stride=1, padding=2)  # 5x5最大池化
        x_2 = torch.nn.functional.max_pool2d(x, 9, stride=1, padding=4)  # 9x9最大池化
        x_3 = torch.nn.functional.max_pool2d(x, 13, stride=1, padding=6) # 13x13最大池化
        x = torch.cat([x, x_1, x_2, x_3], dim=1)  # 在通道维度拼接

        return x


class ModelEMA(object):
    """模型指数移动平均
    
    Args:
        model: 需要进行EMA的模型
        decay (float): 衰减率
        updates (int): 更新次数
    """
    def __init__(self, model, decay=0.9999, updates=0):
        # 创建EMA模型
        self.ema = deepcopy(model).eval()  # 复制模型并设为评估模式
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000.))  # 动态衰减率
        for p in self.ema.parameters():
            p.requires_grad_(False)  # 冻结EMA模型参数

    def update(self, model):
        """更新EMA模型参数"""
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)  # 获取当前衰减率

            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:  # 只更新浮点类型参数
                    v *= d  # EMA更新公式
                    v += (1. - d) * msd[k].detach()
