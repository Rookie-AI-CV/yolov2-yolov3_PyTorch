import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


# 指定可被其他模块导入的函数名
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


# 预训练模型的URL地址
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """带填充的3x3卷积"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1卷积"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    """ResNet基本残差块
    
    Args:
        inplanes (int): 输入通道数
        planes (int): 输出通道数
        stride (int): 步长
        downsample (nn.Module): 下采样层
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    """ResNet瓶颈残差块
    
    Args:
        inplanes (int): 输入通道数
        planes (int): 中间层通道数
        stride (int): 步长
        downsample (nn.Module): 下采样层
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """ResNet骨干网络
    
    Args:
        block (nn.Module): 残差块类型(BasicBlock或Bottleneck)
        layers (list): 每个阶段的残差块数量
        zero_init_residual (bool): 是否将残差块最后的BN层初始化为0
    """
    def __init__(self, block, layers, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        # 第一个卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False) # 输入通道数为3，输出通道数为64，卷积核大小为7，步长为2，填充为3，不使用偏置
        self.bn1 = nn.BatchNorm2d(64) # 批归一化层，输入通道数为64
        self.relu = nn.ReLU(inplace=True) # ReLU激活函数
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 最大池化层，池化窗口大小为3，步长为2，填充为1
        # 四个残差层
        self.layer1 = self._make_layer(block, 64, layers[0]) # 第一个残差层，输入通道数为64，输出通道数为64，残差块数量为layers[0]
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # 第二个残差层，输入通道数为128，输出通道数为128，残差块数量为layers[1]，步长为2
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) # 第三个残差层，输入通道数为256，输出通道数为256，残差块数量为layers[2]，步长为2
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2) # 第四个残差层，输入通道数为512，输出通道数为512，残差块数量为layers[3]，步长为2

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # 使用Kaiming初始化方法初始化卷积层的权重
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1) # 将BN层的权重初始化为1
                nn.init.constant_(m.bias, 0) # 将BN层的偏置初始化为0    

        # 将残差块最后的BN层初始化为0,使每个残差块的初始状态为恒等映射
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0) # 将瓶颈层的BN层的权重初始化为0
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0) # 将基本层的BN层的权重初始化为0

    def _make_layer(self, block, planes, blocks, stride=1):
        """构建残差层
        
        Args:
            block (nn.Module): 残差块类型
            planes (int): 输出通道数
            blocks (int): 残差块数量
            stride (int): 步长
            
        Returns:
            nn.Sequential: 残差层
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion: # 如果步长不为1或者输入通道数不等于输出通道数的4倍
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride), # 使用1x1卷积进行下采样
                nn.BatchNorm2d(planes * block.expansion), # 使用批归一化层
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample)) # 添加第一个残差块
        self.inplanes = planes * block.expansion # 更新输入通道数
        for _ in range(1, blocks): # 添加剩余的残差块
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

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
        c1 = self.conv1(x) # 第一个卷积层
        c1 = self.bn1(c1) # 批归一化层
        c1 = self.relu(c1) # ReLU激活函数
        c1 = self.maxpool(c1) # 最大池化层

        c2 = self.layer1(c1) # 第一个残差层
        c3 = self.layer2(c2) # 第二个残差层
        c4 = self.layer3(c3) # 第三个残差层
        c5 = self.layer4(c4) # 第四个残差层

        output = {
            'layer1': c3,          # 8倍下采样
            'layer2': c4,          # 16倍下采样
            'layer3': c5           # 32倍下采样
        }

        return output


def resnet18(pretrained=False, **kwargs):
    """构建ResNet-18模型
    
    Args:
        pretrained (bool): 是否加载预训练权重
        
    Returns:
        model: 构建好的模型
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        # strict=False因为我们不需要fc层的参数
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model

def resnet34(pretrained=False, **kwargs):
    """构建ResNet-34模型
    
    Args:
        pretrained (bool): 是否加载预训练权重
        
    Returns:
        model: 构建好的模型
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
    return model

def resnet50(pretrained=False, **kwargs):
    """构建ResNet-50模型
    
    Args:
        pretrained (bool): 是否加载预训练权重
        
    Returns:
        model: 构建好的模型
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model

def resnet101(pretrained=False, **kwargs):
    """构建ResNet-101模型
    
    Args:
        pretrained (bool): 是否加载预训练权重
        
    Returns:
        model: 构建好的模型
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
    return model

def resnet152(pretrained=False, **kwargs):
    """构建ResNet-152模型
    
    Args:
        pretrained (bool): 是否加载预训练权重
        
    Returns:
        model: 构建好的模型
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


def build_resnet(model_name='resnet18', pretrained=False):
    """构建ResNet模型的工厂函数
    
    Args:
        model_name (str): 模型名称
        pretrained (bool): 是否加载预训练权重
        
    Returns:
        model: 构建好的模型
    """
    if model_name == 'resnet18':
        model = resnet18(pretrained=pretrained)
    
    elif model_name == 'resnet34':
        model = resnet34(pretrained=pretrained)
    
    elif model_name == 'resnet50':
        model = resnet50(pretrained=pretrained)
    
    elif model_name == 'resnet101':
        model = resnet101(pretrained=pretrained)

    elif model_name == 'resnet152':
        model = resnet152(pretrained=pretrained)
    

    return model


if __name__ == "__main__":
    import time

    model = build_resnet(model_name='resnet18', pretrained=True)
    x = torch.randn(1, 3, 224, 224)
    t0 = time.time()
    output = model(x)
    t1 = time.time()
    print('Time: ', t1 - t0)

    for k in output.keys():
        print('{} : {}'.format(k, output[k].shape))
