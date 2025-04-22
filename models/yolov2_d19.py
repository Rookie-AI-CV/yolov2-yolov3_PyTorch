import numpy as np
import torch
import torch.nn as nn
from utils.modules import Conv, reorg_layer

from backbone import build_backbone
import tools


class YOLOv2D19(nn.Module):
    """YOLOv2 使用Darknet19作为骨干网络的目标检测模型
    
    Args:
        device: 运行设备(CPU/GPU)
        input_size: 输入图像尺寸
        num_classes: 类别数量
        trainable: 是否为训练模式
        conf_thresh: 置信度阈值
        nms_thresh: NMS阈值
        anchor_size: 预设的anchor尺寸
    """
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.001, nms_thresh=0.5, anchor_size=None):
        super(YOLOv2D19, self).__init__()
        self.device = device
        self.input_size = input_size    # 输入图像尺寸
        self.num_classes = num_classes  # 类别数量
        self.trainable = trainable      # 是否为训练模式
        self.conf_thresh = conf_thresh  # 置信度阈值
        self.nms_thresh = nms_thresh    # NMS阈值
        self.anchor_size = torch.tensor(anchor_size)  # 预设的anchor尺寸
        self.num_anchors = len(anchor_size)  # anchor数量
        self.stride = 32  # 特征图相对于原图的下采样倍数
        self.grid_cell, self.all_anchor_wh = self.create_grid(input_size)  # 创建网格点和anchor尺寸张量 

        # 骨干网络: darknet-19
        self.backbone = build_backbone(model_name='darknet19', pretrained=trainable)
        
        # 检测头部分
        self.convsets_1 = nn.Sequential(
            Conv(1024, 1024, k=3, p=1),
            Conv(1024, 1024, k=3, p=1)
        )

        # 路由层和重组层
        self.route_layer = Conv(512, 64, k=1)
        self.reorg = reorg_layer(stride=2)

        self.convsets_2 = Conv(1280, 1024, k=3, p=1)
        
        # 预测层: 输出通道数 = anchor数量 * (1个置信度 + 4个边框参数 + 类别数)
        self.pred = nn.Conv2d(1024, self.num_anchors*(1 + 4 + self.num_classes), kernel_size=1)


    def create_grid(self, input_size):
        """创建网格点和anchor尺寸张量
        
        Args:
            input_size: 输入图像尺寸
            
        Returns:
            grid_xy: 网格点坐标
            anchor_wh: anchor的宽高
        """
        w, h = input_size, input_size
        # 生成网格点
        ws, hs = w // self.stride, h // self.stride
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        grid_xy = grid_xy.view(1, hs*ws, 1, 2).to(self.device)

        # 生成anchor宽高张量
        anchor_wh = self.anchor_size.repeat(hs*ws, 1, 1).unsqueeze(0).to(self.device)

        return grid_xy, anchor_wh


    def set_grid(self, input_size):
        """重新设置网格点和anchor尺寸
        
        Args:
            input_size: 新的输入图像尺寸
        """
        self.input_size = input_size
        self.grid_cell, self.all_anchor_wh = self.create_grid(input_size)


    def decode_xywh(self, txtytwth_pred):
        """将网络预测的tx,ty,tw,th解码为中心点坐标和宽高
        
        Args:
            txtytwth_pred: [B, H*W, anchor_n, 4] 预测的tx,ty,tw,th
            
        Returns:
            xywh_pred: [B, H*W*anchor_n, 4] 解码后的中心点坐标和宽高
        """
        B, HW, ab_n, _ = txtytwth_pred.size()
        # b_x = sigmoid(tx) + gride_x
        # b_y = sigmoid(ty) + gride_y
        xy_pred = torch.sigmoid(txtytwth_pred[..., :2]) + self.grid_cell
        # b_w = anchor_w * exp(tw)
        # b_h = anchor_h * exp(th)
        wh_pred = torch.exp(txtytwth_pred[..., 2:]) * self.all_anchor_wh
        # [B, H*W, anchor_n, 4] -> [B, H*W*anchor_n, 4]
        xywh_pred = torch.cat([xy_pred, wh_pred], -1).view(B, -1, 4) * self.stride

        return xywh_pred
    

    def decode_boxes(self, txtytwth_pred):
        """将网络预测的tx,ty,tw,th解码为边界框坐标
        
        Args:
            txtytwth_pred: [B, H*W, anchor_n, 4] 预测的tx,ty,tw,th
            
        Returns:
            x1y1x2y2_pred: [B, H*W*anchor_n, 4] 解码后的边界框坐标(x1,y1,x2,y2)
        """
        # txtytwth -> cxcywh
        xywh_pred = self.decode_xywh(txtytwth_pred)

        # cxcywh -> x1y1x2y2
        x1y1x2y2_pred = torch.zeros_like(xywh_pred)
        x1y1_pred = xywh_pred[..., :2] - xywh_pred[..., 2:] * 0.5
        x2y2_pred = xywh_pred[..., :2] + xywh_pred[..., 2:] * 0.5
        x1y1x2y2_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)
        
        return x1y1x2y2_pred


    def nms(self, dets, scores):
        """非极大值抑制
        
        Args:
            dets: 边界框坐标
            scores: 对应的分数
            
        Returns:
            keep: 保留的边界框索引
        """
        x1 = dets[:, 0]  #xmin
        y1 = dets[:, 1]  #ymin
        x2 = dets[:, 2]  #xmax
        y2 = dets[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]  # 按分数从大到小排序

        keep = []
        while order.size > 0:
            i = order[0]  # 取分数最大的边界框
            keep.append(i)
            # 计算其他边界框与当前边界框的IoU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-10, xx2 - xx1)
            h = np.maximum(1e-10, yy2 - yy1)
            inter = w * h

            # 计算IoU: 交集面积/(框1面积 + 框2面积 - 交集面积)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # 保留IoU小于阈值的边界框
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep


    def postprocess(self, bboxes, scores):
        """后处理: 对预测结果进行筛选
        
        Args:
            bboxes: (HxW, 4) 预测的边界框
            scores: (HxW, num_classes) 预测的分数
            
        Returns:
            bboxes: 筛选后的边界框
            scores: 筛选后的分数
            cls_inds: 筛选后的类别索引
        """
        # 获取每个边界框的最高分数对应的类别
        cls_inds = np.argmax(scores, axis=1)
        scores = scores[(np.arange(scores.shape[0]), cls_inds)]
        
        # 置信度阈值筛选
        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        # 对每个类别分别进行NMS
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        return bboxes, scores, cls_inds


    @torch.no_grad()
    def inference(self, x):
        """模型推理
        
        Args:
            x: 输入图像
            
        Returns:
            bboxes: 预测的边界框
            scores: 预测的分数
            cls_inds: 预测的类别
        """
        # 骨干网络特征提取
        feats = self.backbone(x)

        # 特征融合
        p5 = self.convsets_1(feats['layer3'])
        p4 = self.reorg(self.route_layer(feats['layer2']))
        p5 = torch.cat([p4, p5], dim=1)

        # 检测头
        p5 = self.convsets_2(p5)

        # 预测
        pred = self.pred(p5)

        B, abC, H, W = pred.size()

        # 调整预测结果维度
        # [B, num_anchor * C, H, W] -> [B, H, W, num_anchor * C] -> [B, H*W, num_anchor*C]
        pred = pred.permute(0, 2, 3, 1).contiguous().view(B, H*W, abC)

        # [B, H*W*num_anchor, 1] 置信度预测
        conf_pred = pred[:, :, :1 * self.num_anchors].contiguous().view(B, H*W*self.num_anchors, 1)
        # [B, H*W, num_anchor, num_cls] 类别预测
        cls_pred = pred[:, :, 1 * self.num_anchors : (1 + self.num_classes) * self.num_anchors].contiguous().view(B, H*W*self.num_anchors, self.num_classes)
        # [B, H*W, num_anchor, 4] 边界框预测
        reg_pred = pred[:, :, (1 + self.num_classes) * self.num_anchors:].contiguous()
        reg_pred = reg_pred.view(B, H*W, self.num_anchors, 4)
        box_pred = self.decode_boxes(reg_pred)

        # 只处理batch size = 1的情况
        conf_pred = conf_pred[0]
        cls_pred = cls_pred[0]
        box_pred = box_pred[0]

        # 计算最终得分
        scores = torch.sigmoid(conf_pred) * torch.softmax(cls_pred, dim=-1)

        # 将边界框坐标归一化到[0,1]
        bboxes = torch.clamp(box_pred / self.input_size, 0., 1.)

        # 转移到CPU并转换为numpy数组
        scores = scores.to('cpu').numpy()
        bboxes = bboxes.to('cpu').numpy()

        # 后处理
        bboxes, scores, cls_inds = self.postprocess(bboxes, scores)

        return bboxes, scores, cls_inds


    def forward(self, x, target=None):
        """前向传播
        
        Args:
            x: 输入图像
            target: 训练目标
            
        Returns:
            如果是推理模式:
                返回预测的边界框、分数和类别
            如果是训练模式:
                返回各个损失项(置信度损失、类别损失、边界框损失、IoU损失)
        """
        if not self.trainable:
            return self.inference(x)
        else:
            # 骨干网络特征提取
            feats = self.backbone(x)

            # 特征融合
            p5 = self.convsets_1(feats['layer3'])
            p4 = self.reorg(self.route_layer(feats['layer2']))
            p5 = torch.cat([p4, p5], dim=1)

            # 检测头
            p5 = self.convsets_2(p5)

            # 预测
            pred = self.pred(p5)

            B, abC, H, W = pred.size()

            # 调整预测结果维度
            pred = pred.permute(0, 2, 3, 1).contiguous().view(B, H*W, abC)

            # 分离预测结果
            # 置信度预测
            conf_pred = pred[:, :, :1 * self.num_anchors].contiguous().view(B, H*W*self.num_anchors, 1)
            # 类别预测
            cls_pred = pred[:, :, 1 * self.num_anchors : (1 + self.num_classes) * self.num_anchors].contiguous().view(B, H*W*self.num_anchors, self.num_classes)
            # 边界框预测
            reg_pred = pred[:, :, (1 + self.num_classes) * self.num_anchors:].contiguous()
            reg_pred = reg_pred.view(B, H*W, self.num_anchors, 4)

            # 解码边界框并计算与真实框的IoU
            x1y1x2y2_pred = (self.decode_boxes(reg_pred) / self.input_size).view(-1, 4)
            x1y1x2y2_gt = target[:, :, 7:].view(-1, 4)
            reg_pred = reg_pred.view(B, H*W*self.num_anchors, 4)

            # 计算置信度目标
            iou_pred = tools.iou_score(x1y1x2y2_pred, x1y1x2y2_gt).view(B, -1, 1)
            gt_conf = iou_pred.clone().detach()

            # 组合训练目标
            # [obj, cls, txtytwth, x1y1x2y2] -> [conf, obj, cls, txtytwth]
            target = torch.cat([gt_conf, target[:, :, :7]], dim=2)

            # 计算损失
            (
                conf_loss,  # 置信度损失
                cls_loss,   # 类别损失
                bbox_loss,  # 边界框损失
                iou_loss    # IoU损失
            ) = tools.loss(pred_conf=conf_pred,
                           pred_cls=cls_pred,
                           pred_txtytwth=reg_pred,
                           pred_iou=iou_pred,
                           label=target
                           )

            return conf_loss, cls_loss, bbox_loss, iou_loss   
