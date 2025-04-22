import torch
import torch.nn as nn


# 预训练模型的URL地址
model_urls = {
    "darknet_tiny": "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/darknet_tiny.pth",
}


# 指定可被其他模块导入的函数名
__all__ = ['darknet_tiny']


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


class DarkNet_Tiny(nn.Module):
    """轻量级DarkNet骨干网络
    
    网络结构包含7个卷积层和6个最大池化层,用于特征提取
    """
    def __init__(self):
        super(DarkNet_Tiny, self).__init__()
        # 骨干网络 : DarkNet_Tiny
        # 第一个卷积块: 3->16
        self.conv_1 = Conv_BN_LeakyReLU(3, 16, 3, 1)
        self.maxpool_1 = nn.MaxPool2d((2, 2), 2)              # 下采样2倍

        # 第二个卷积块: 16->32
        self.conv_2 = Conv_BN_LeakyReLU(16, 32, 3, 1)
        self.maxpool_2 = nn.MaxPool2d((2, 2), 2)              # 下采样4倍

        # 第三个卷积块: 32->64
        self.conv_3 = Conv_BN_LeakyReLU(32, 64, 3, 1)
        self.maxpool_3 = nn.MaxPool2d((2, 2), 2)              # 下采样8倍

        # 第四个卷积块: 64->128
        self.conv_4 = Conv_BN_LeakyReLU(64, 128, 3, 1)
        self.maxpool_4 = nn.MaxPool2d((2, 2), 2)              # 下采样16倍

        # 第五个卷积块: 128->256
        self.conv_5 = Conv_BN_LeakyReLU(128, 256, 3, 1)
        self.maxpool_5 = nn.MaxPool2d((2, 2), 2)              # 下采样32倍

        # 第六个卷积块: 256->512
        self.conv_6 = Conv_BN_LeakyReLU(256, 512, 3, 1)
        self.maxpool_6 = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),                       # 右下各填充1个像素
            nn.MaxPool2d((2, 2), 1)                           # 保持32倍下采样
        )

        # 第七个卷积块: 512->1024
        self.conv_7 = Conv_BN_LeakyReLU(512, 1024, 3, 1)


    def forward(self, x):
        """前向传播函数
        
        Args:
            x: 输入张量
            
        Returns:
            dict: 包含三个特征层的字典
                - layer1: 8倍下采样的特征图
                - layer2: 16倍下采样的特征图  
                - layer3: 32倍下采样的特征图
        """
        x = self.conv_1(x)
        c1 = self.maxpool_1(x)
        c1 = self.conv_2(c1)
        c2 = self.maxpool_2(c1)
        c2 = self.conv_3(c2)
        c3 = self.maxpool_3(c2)
        c3 = self.conv_4(c3)
        c4 = self.maxpool_4(c3)
        c4 = self.conv_5(c4)       # 16倍下采样
        c5 = self.maxpool_5(c4)  
        c5 = self.conv_6(c5)
        c5 = self.maxpool_6(c5)
        c5 = self.conv_7(c5)       # 32倍下采样

        output = {
            'layer1': c3,          # 8倍下采样
            'layer2': c4,          # 16倍下采样
            'layer3': c5           # 32倍下采样
        }

        return output


def build_darknet_tiny(pretrained=False):
    """构建DarkNet-Tiny模型
    
    Args:
        pretrained (bool): 是否加载预训练权重
        
    Returns:
        model: 构建好的模型
    """
    # 创建模型
    model = DarkNet_Tiny()

    # 加载预训练权重
    if pretrained:
        print('Loading pretrained weight ...')
        url = model_urls['darknet_tiny']
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
    # 测试代码
    net = build_darknet_tiny(pretrained=True)
    x = torch.randn(1, 3, 224, 224)
    t0 = time.time()
    output = net(x)
    t1 = time.time()
    print('Time: ', t1 - t0)

    for k in output.keys():
        print('{} : {}'.format(k, output[k].shape))
