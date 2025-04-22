import cv2
import numpy as np
from numpy import random


def intersect(box_a, box_b):
    """计算两个框的交集面积"""
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """计算两组框的Jaccard重叠度。
    Jaccard重叠度就是两个框的交集除以并集。
    例如:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    参数:
        box_a: 多个边界框, 形状: [num_boxes,4]
        box_b: 单个边界框, 形状: [4]
    返回:
        jaccard重叠度: 形状: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """将多个数据增强操作组合在一起。
    参数:
        transforms (List[Transform]): 要组合的变换列表。
    示例:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class ConvertFromInts(object):
    """将图像从整数类型转换为浮点类型"""
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class Normalize(object):
    """图像标准化处理"""
    def __init__(self, mean=None, std=None):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image /= 255.
        image -= self.mean
        image /= self.std

        return image, boxes, labels


class ToAbsoluteCoords(object):
    """将归一化的坐标转换为绝对坐标"""
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords(object):
    """将绝对坐标转换为归一化坐标"""
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class Resize(object):
    """调整图像大小"""
    def __init__(self, size=416):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size, self.size))
        return image, boxes, labels


class RandomSaturation(object):
    """随机调整饱和度"""
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "对比度上限必须大于等于下限"
        assert self.lower >= 0, "对比度下限必须非负"

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels


class RandomHue(object):
    """随机调整色调"""
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomLightingNoise(object):
    """随机调整光照噪声"""
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # 打乱通道
            image = shuffle(image)
        return image, boxes, labels


class ConvertColor(object):
    """颜色空间转换"""
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomContrast(object):
    """随机调整对比度"""
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "对比度上限必须大于等于下限"
        assert self.lower >= 0, "对比度下限必须非负"

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class RandomBrightness(object):
    """随机调整亮度"""
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class RandomSampleCrop(object):
    """随机裁剪
    参数:
        img (Image): 训练时输入的图像
        boxes (Tensor): 原始边界框(pt格式)
        labels (Tensor): 每个边界框的类别标签
        mode (float tuple): 最小和最大jaccard重叠度
    返回:
        (img, boxes, classes)
            img (Image): 裁剪后的图像
            boxes (Tensor): 调整后的边界框(pt格式)
            labels (Tensor): 每个边界框的类别标签
    """
    def __init__(self):
        self.sample_options = (
            # 使用整个原始输入图像
            None,
            # 采样一个补丁使得与目标的最小jaccard重叠度为.1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # 随机采样一个补丁
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # 随机选择一个模式
            sample_id = np.random.randint(len(self.sample_options))
            mode = self.sample_options[sample_id]
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # 最大尝试次数(50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # 宽高比约束在0.5到2之间
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # 转换为整数矩形坐标 x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # 计算裁剪框与真实框之间的IoU(jaccard重叠度)
                overlap = jaccard_numpy(boxes, rect)

                # 是否满足最小和最大重叠度约束,不满足则重试
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # 从图像中裁剪
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # 保留中心点在采样补丁内的真实框
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # 掩码标记所有中心点在左上方的真实框
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # 掩码标记所有中心点在右下方的真实框
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # 掩码标记同时满足m1和m2的真实框
                mask = m1 * m2

                # 如果没有有效框则重试
                if not mask.any():
                    continue

                # 取出匹配的真实框
                current_boxes = boxes[mask, :].copy()

                # 取出匹配的标签
                current_labels = labels[mask]

                # 调整真实框坐标
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # 调整到裁剪区域(减去裁剪区域的左上角坐标)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # 调整到裁剪区域(减去裁剪区域的左上角坐标)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels


class RandomMirror(object):
    """随机水平翻转"""
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes


class SwapChannels(object):
    """按指定顺序交换图像的通道
    参数:
        swaps (int triple): 通道的最终顺序
            例如: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        参数:
            image (Tensor): 要变换的图像张量
        返回:
            按照swap交换通道后的张量
        """
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    """光度失真"""
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return im, boxes, labels


class SSDAugmentation(object):
    """SSD数据增强"""
    def __init__(self, size=416, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
        self.mean = mean
        self.size = size
        self.std = std
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            PhotometricDistort(),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            Normalize(self.mean, self.std)
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)


class ColorAugmentation(object):
    """颜色增强"""
    def __init__(self, size=416, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
        self.mean = mean
        self.size = size
        self.std = std
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            PhotometricDistort(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            Normalize(self.mean, self.std)
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)
