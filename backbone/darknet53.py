import torch
import torch.nn as nn


# 预训练模型的URL地址
model_urls = {
    "darknet53": "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/darknet53.pth",
}


# 指定可被其他模块导入的函数名
__all__ = ['darknet53']


class Conv_BN_LeakyReLU(nn.Module):
    """卷积-批归一化-LeakyReLU激活函数组合模块
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数 
        ksize (int): 卷积核大小
        padding (int): 填充大小
        stride (int): 步长
        dilation (int): 膨胀系数
    """
    def __init__(self, in_channels, out_channels, ksize, padding=0, stride=1, dilation=1):
        super(Conv_BN_LeakyReLU, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, padding=padding, stride=stride, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.convs(x)


class ResBlock(nn.Module):
    """残差块模块
    
    Args:
        ch (int): 输入输出通道数
        nblocks (int): 残差块重复次数
    """
    def __init__(self, ch, nblocks=1):
        super().__init__()
        self.module_list = nn.ModuleList()
        for _ in range(nblocks):
            resblock_one = nn.Sequential(
                Conv_BN_LeakyReLU(ch, ch//2, 1),           # 1x1卷积降维
                Conv_BN_LeakyReLU(ch//2, ch, 3, padding=1) # 3x3卷积特征提取
            )
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            x = module(x) + x    # 残差连接
        return x


class DarkNet_53(nn.Module):
    """DarkNet-53骨干网络
    
    网络结构包含5个大的卷积块,每个块内包含多个残差块,用于特征提取。
    总共53个卷积层,故名DarkNet-53。
    """
    def __init__(self):
        super(DarkNet_53, self).__init__()
        # 第一个卷积块: 2倍下采样
        self.layer_1 = nn.Sequential(
            Conv_BN_LeakyReLU(3, 32, 3, padding=1),
            Conv_BN_LeakyReLU(32, 64, 3, padding=1, stride=2),
            ResBlock(64, nblocks=1)
        )
        # 第二个卷积块: 4倍下采样
        self.layer_2 = nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, 3, padding=1, stride=2),
            ResBlock(128, nblocks=2)
        )
        # 第三个卷积块: 8倍下采样
        self.layer_3 = nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, 3, padding=1, stride=2),
            ResBlock(256, nblocks=8)
        )
        # 第四个卷积块: 16倍下采样
        self.layer_4 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, 3, padding=1, stride=2),
            ResBlock(512, nblocks=8)
        )
        # 第五个卷积块: 32倍下采样
        self.layer_5 = nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, 3, padding=1, stride=2),
            ResBlock(1024, nblocks=4)
        )


    def forward(self, x, targets=None):
        """前向传播函数
        
        Args:
            x: 输入张量
            targets: 目标值(训练时使用)
            
        Returns:
            dict: 包含三个特征层的字典
                - layer1: 8倍下采样的特征图
                - layer2: 16倍下采样的特征图  
                - layer3: 32倍下采样的特征图
        """
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        output = {
            'layer1': c3,          # 8倍下采样
            'layer2': c4,          # 16倍下采样
            'layer3': c5           # 32倍下采样
        }

        return output


def build_darknet53(pretrained=False):
    """构建DarkNet-53模型
    
    Args:
        pretrained (bool): 是否加载预训练权重
        
    Returns:
        model: 构建好的模型
    """
    # 创建模型
    model = DarkNet_53()

    # 加载预训练权重
    if pretrained:
        print('Loading pretrained weight ...')
        url = model_urls['darknet53']
        # 下载预训练权重
        checkpoint_state_dict = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu", check_hash=True)
        # 获取模型当前权重
        model_state_dict = model.state_dict()
        # 检查权重维度匹配
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    checkpoint_state_dict.pop(k)
            else:
                checkpoint_state_dict.pop(k)
                print(k)

        model.load_state_dict(checkpoint_state_dict)

    return model


if __name__ == '__main__':
    import time
    net = build_darknet53(pretrained=True)
    x = torch.randn(1, 3, 224, 224)
    t0 = time.time()
    output = net(x)
    t1 = time.time()
    print('Time: ', t1 - t0)

    for k in output.keys():
        print('{} : {}'.format(k, output[k].shape))
