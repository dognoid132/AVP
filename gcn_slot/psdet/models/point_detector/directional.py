import math

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch
import torch.nn.functional as F
import kornia
from .featfusion import MultiscaleFusionWithAttention
from .gcn import GCNEncoder, EdgePredictor
from .post_process import calc_point_squre_dist, pass_through_third_point
from .post_process import get_predicted_points, get_predicted_directional_points, pair_marking_points
from .utils import define_halve_unit, define_detector_block, YetAnotherDarknet, vgg16, resnet18, resnet50


def enhance_lane_lines(img):
    """
    增强 BEV 图像中的白色和黄色车道线，抑制黑区，保留真实色彩，并叠加Canny边缘
    输入: BGR uint8 图像 (H, W, 3)
    输出: 增强后的 BGR uint8 图像
    """
    # 1. 转换到 HSV 空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 2. 提取白色车道线（高亮度，低饱和）
    white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 30, 255))

    # 3. 提取黄色车道线（H ～ 30～40, S 高, V 中高）
    yellow_lower = np.array([20, 70, 50])
    yellow_upper = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    # 4. 合并白色+黄色掩码
    lane_mask = cv2.bitwise_or(white_mask, yellow_mask)

    # 5. 使用 Canny 检测边缘（用于强化细线）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges_canny = cv2.Canny(gray, 50, 150)
    # edges_canny = cv2.dilate(edges_canny, None)  # 扩展边缘，防止太细

    # # 使用 Sobel 算子提取边缘
    # sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)  # 水平方向
    # sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)  # 垂直方向
    # sobel = cv2.addWeighted(cv2.convertScaleAbs(sobel_x), 0.5, cv2.convertScaleAbs(sobel_y), 0.5, 0)
    #
    # # 合并 Canny 边缘和 Sobel 边缘
    # edges = cv2.bitwise_or(edges_canny, sobel)

    # 6. 合并：车道线 + 边缘 → 更精确的标线区域
    combined_mask = cv2.bitwise_or(lane_mask, edges_canny)

    # 7. 对原图进行 CLAHE（仅对 L 通道）
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 8. 局部增强：只在标线区域提高亮度，不改变颜色
    # 创建一个“亮度提升”层（保持原色）
    overlay = img_clahe.copy()

    # 在 mask 区域内，增加亮度（不直接叠加白色）
    overlay[lane_mask > 0] = cv2.addWeighted(
        img_clahe[lane_mask > 0], 0.7,
        np.full_like(img_clahe[lane_mask > 0], 255), 0.3,
        0
    ).astype(np.uint8)

    # 对提取出的白色和黄色区域进行增强（增加亮度和饱和度）
    hsv_overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_overlay)

    # 提升亮度和饱和度
    s[lane_mask > 0] = np.clip(s[lane_mask > 0] * 1.5, 0, 255)
    v[lane_mask > 0] = np.clip(v[lane_mask > 0] * 1.2 + 20, 0, 255)
    enhanced_hsv = cv2.merge([h, s, v])
    overlay = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

    # 9. 抑制黑色块：对暗区轻微提亮
    gray_img = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2GRAY)
    _, dark_mask = cv2.threshold(gray_img, 30, 255, cv2.THRESH_BINARY_INV)
    dark_mask = cv2.erode(dark_mask, None)  # 缩小暗区范围

    result = img_clahe.copy()
    result[dark_mask > 0] = cv2.add(result[dark_mask > 0], (10, 10, 10))

    # 10. 最终结果：将增强后的标线融合，并叠加Canny边缘
    final_result = result.copy()
    final_result[lane_mask > 0] = overlay[lane_mask > 0]

    # # 在最终结果上叠加Canny边缘（用红色表示）
    # edge_color = (0, 0, 255)  # 红色
    # final_result[edges_canny > 0] = edge_color

    return final_result


class PointDetector(nn.modules.Module):
    """Detector for point without direction."""

    def __init__(self, cfg):
        super(PointDetector, self).__init__()
        self.cfg = cfg
        input_channel_size = cfg.input_channels
        depth_factor = cfg.depth_factor
        output_channel_size = cfg.output_channels
        self.use_multi_scale = False
        self.point_loss_func = nn.MSELoss().cuda()

        if cfg.backbone == 'Darknet':
            self.feature_extractor = YetAnotherDarknet(input_channel_size, depth_factor)
        elif cfg.backbone == 'VGG16':
            self.feature_extractor = vgg16()
        elif cfg.backbone == 'resnet18':
            self.feature_extractor = resnet18()
        elif cfg.backbone == 'resnet50':
            self.feature_extractor = resnet50()
        else:
            raise ValueError('{} is not implemented!'.format(cfg.backbone))

        # if cfg.backbone == 'VGG16' and cfg.get('use_multiscale_fusion', False):
        #     self.feature_extractor = vgg16(multi_feature=True)  # 注意开启 multi_feature
        #     self.use_multi_scale = True
        #     self.fusion_module = MultiscaleFusionWithAttention(out_ch=16 * cfg.depth_factor)  # 与 detector 输入通道对齐！
        # else:
        #     self.feature_extractor = ...  # 原始单输出
        #     self.use_multi_scale = False

        layers_points = []
        layers_points += define_detector_block(16 * depth_factor)
        layers_points += define_detector_block(16 * depth_factor)
        layers_points += [nn.Conv2d(32 * depth_factor, output_channel_size,
                                    kernel_size=1, stride=1, padding=0, bias=False)]
        self.point_predictor = nn.Sequential(*layers_points)

        layers_descriptor = []
        layers_descriptor += define_detector_block(16 * depth_factor)
        layers_descriptor += define_detector_block(16 * depth_factor)
        layers_descriptor += [nn.Conv2d(32 * depth_factor, cfg.descriptor_dim,
                                        kernel_size=1, stride=1, padding=0, bias=False)]
        self.descriptor_map = nn.Sequential(*layers_descriptor)

        if cfg.use_gnn:
            self.graph_encoder = GCNEncoder(cfg.graph_encoder)

        self.edge_predictor = EdgePredictor(cfg.edge_predictor)

        if cfg.get('slant_predictor', None):
            self.slant_predictor = EdgePredictor(cfg.slant_predictor)

        if cfg.get('vacant_predictor', None):
            self.vacant_predictor = EdgePredictor(cfg.vacant_predictor)

    # def forward(self, data_dict):
    #     img = data_dict['image']
    #
    #     features = self.feature_extractor(img)  # [b, 1024, 16, 16]
    #
    #     points_pred = self.point_predictor(features)
    #     points_pred = torch.sigmoid(points_pred)
    #     data_dict['points_pred'] = points_pred
    #
    #     descriptor_map = self.descriptor_map(features)
    #
    #     if self.training:
    #         marks = data_dict['marks']
    #         pred_dict = self.predict_slots(descriptor_map, marks[:, :, :2])
    #         data_dict.update(pred_dict)
    #     else:
    #         data_dict['descriptor_map'] = descriptor_map
    #
    #     return data_dict

    def forward(self, data_dict):
        img = data_dict['image']
        if self.cfg.enhance:
            img = enhance_lane_lines(img)
        features = self.feature_extractor(img)

        # 处理多尺度输出
        if self.use_multi_scale:
            feat = self.fusion_module(features['c3'], features['c4'], features['c5'])
        else:
            feat = features  # 原始单输出
        # print("-------------------------------------------------------------------------------")
        points_pred = self.point_predictor(feat)
        points_pred = torch.sigmoid(points_pred)
        data_dict['points_pred'] = points_pred

        descriptor_map = self.descriptor_map(feat)

        if self.training:
            marks = data_dict['marks']
            pred_dict = self.predict_slots(descriptor_map, marks[:, :, :2])
            data_dict.update(pred_dict)
        else:
            data_dict['descriptor_map'] = descriptor_map

        return data_dict

    def predict_slots(self, descriptor_map, points):
        descriptors = self.sample_descriptors(descriptor_map, points)
        data_dict = {}
        data_dict['descriptors'] = descriptors
        data_dict['points'] = points

        if self.cfg.get('slant_predictor', None):
            pred_dict = self.slant_predictor(data_dict)
            data_dict['slant_pred'] = pred_dict['edges_pred']

        if self.cfg.get('vacant_predictor', None):
            pred_dict = self.vacant_predictor(data_dict)
            data_dict['vacant_pred'] = pred_dict['edges_pred']

        if self.cfg.use_gnn:
            data_dict = self.graph_encoder(data_dict)

        pred_dict = self.edge_predictor(data_dict)

        data_dict['edge_pred'] = pred_dict['edges_pred']
        return data_dict

    def sample_descriptors(self, descriptors, keypoints):
        """ Interpolate descriptors at keypoint locations """
        b, c, h, w = descriptors.shape
        keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
        args = {'align_corners': True} if int(torch.__version__[2]) > 2 else {}
        descriptors = torch.nn.functional.grid_sample(
            descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
        descriptors = torch.nn.functional.normalize(
            descriptors.reshape(b, c, -1), p=2, dim=1)
        return descriptors

    def get_targets_points(self, data_dict):
        points_pred = data_dict['points_pred']
        marks_gt_batch = data_dict['marks']
        npoints = data_dict['npoints']

        b, c, h, w = points_pred.shape
        targets = torch.zeros(b, c, h, w).cuda()
        mask = torch.zeros_like(targets)
        mask[:, 0].fill_(1.)

        for batch_idx, marks_gt in enumerate(marks_gt_batch):
            n = npoints[batch_idx].long()
            for marking_point in marks_gt[:n]:
                x, y = marking_point[:2]
                col = math.floor(x * w)
                row = math.floor(y * h)
                # Confidence Regression
                targets[batch_idx, 0, row, col] = 1.
                # Offset Regression
                targets[batch_idx, 1, row, col] = x * w - col
                targets[batch_idx, 2, row, col] = y * h - row

                mask[batch_idx, 1:3, row, col].fill_(1.)
        return targets, mask

    def post_processing(self, data_dict):
        ret_dicts = {}
        pred_dicts = {}

        points_pred = data_dict['points_pred']
        descriptor_map = data_dict['descriptor_map']

        points_pred_batch = []
        slots_pred = []
        for b, marks in enumerate(points_pred):
            points_pred = get_predicted_points(marks, self.cfg.point_thresh, self.cfg.boundary_thresh)
            points_pred_batch.append(points_pred)

            if len(points_pred) > 0:
                points_np = np.concatenate([p[1].reshape(1, -1) for p in points_pred], axis=0)
            else:
                points_np = np.zeros((self.cfg.max_points, 2))

            if points_np.shape[0] < self.cfg.max_points:
                points_full = np.zeros((self.cfg.max_points, 2))
                points_full[:len(points_pred)] = points_np
            else:
                points_full = points_np

            pred_dict = self.predict_slots(descriptor_map[b].unsqueeze(0),
                                           torch.Tensor(points_full).unsqueeze(0).cuda())
            edges = pred_dict['edges_pred'][0]
            n = points_np.shape[0]
            m = points_full.shape[0]

            slots = []
            for i in range(n):
                for j in range(n):
                    idx = i * m + j
                    score = edges[0, idx]
                    if score > 0.5:
                        x1, y1 = points_np[i, :2]
                        x2, y2 = points_np[j, :2]
                        slot = (score, np.array([x1, y1, x2, y2]))
                        slots.append(slot)

            slots_pred.append(slots)

        pred_dicts['points_pred'] = points_pred_batch
        pred_dicts['slots_pred'] = slots_pred
        return pred_dicts, ret_dicts

    def compute_canny_distance_weight(images, sigma=5.0, max_dist=20.0):
        """
        计算每张图像的 Canny 边缘距离权重图。

        Args:
            images: (B, 3, H, W), float32, range [0, 1]
            sigma: 控制权重衰减范围（单位：像素）
            max_dist: 距离截断阈值，超过则视为无穷远

        Returns:
            weight: (B, 1, H, W), float32, ∈ (0, 1]
        """
        # Step 1: Canny edge detection (returns (magnitude, edges))
        # edges: (B, 1, H, W), binary {0, 1}
        _, edges = kornia.filters.canny(images, low_threshold=0.1, high_threshold=0.2)

        # Step 2: Compute distance transform to nearest edge
        # Distance is small near edges, large far away
        # Note: distance_transform expects foreground=1, but we want distance to edge (which is 1)
        # So we compute distance to (1 - edges) background? -> No.
        # Instead: use distance to edge mask directly via signed distance or complement trick.
        # Kornia's distance_transform computes distance to foreground (non-zero).
        dist = kornia.contrib.distance_transform(edges)  # (B, 1, H, W)

        # Clamp for stability and control influence radius
        dist = torch.clamp(dist, max=max_dist)

        # Step 3: Exponential decay weight: close to edge → high weight
        weight = torch.exp(-dist / sigma)  # (B, 1, H, W)

        return weight

    def get_training_loss(self, data_dict):
        points_pred = data_dict['points_pred']
        targets, mask = self.get_targets_points(data_dict)

        disp_dict = {}

        loss_point = self.point_loss_func(points_pred * mask, targets * mask)
        if self.cfg.enhance:
            """
             带 Canny 边缘距离加权的训练损失。
             """
            points_pred = data_dict['points_pred']  # (B, 1, H, W)
            targets, mask = self.get_targets_points(data_dict)  # targets: (B,1,H,W), mask: (B,1,H,W)

            # --- Step 1: Compute Canny-based spatial weight ---
            # Assume original input images are in data_dict['images'], shape (B, 3, H_img, W_img)
            images = data_dict['images']  # Must be in [0, 1], float32

            # Resize images to match heatmap resolution if needed
            if images.shape[-2:] != points_pred.shape[-2:]:
                images_resized = F.interpolate(
                    images, size=points_pred.shape[-2:], mode='bilinear', align_corners=False
                )
            else:
                images_resized = images

            canny_weight = self.compute_canny_distance_weight(images_resized, sigma=5, max_dist=20)
            # Ensure same device
            canny_weight = canny_weight.to(points_pred.device)

            # --- Step 2: Point Loss with Canny weighting ---
            # Assume self.point_loss_func supports reduction='none'
            # If it's BCEWithLogitsLoss, make sure it's initialized with reduction='none'
            point_loss_unreduced = self.point_loss_func(points_pred, targets)  # (B, 1, H, W)

            # Apply mask and Canny weight
            weighted_point_loss = point_loss_unreduced * mask * canny_weight
            denominator = (mask * canny_weight).sum() + 1e-6
            loss_point = weighted_point_loss.sum() / denominator
        edges_pred = data_dict['edges_pred']
        edges_target = torch.zeros_like(edges_pred)
        edges_mask = torch.zeros_like(edges_pred)

        match_targets = data_dict['match_targets']
        npoints = data_dict['npoints']

        for b in range(edges_pred.shape[0]):
            n = npoints[b].long()
            y = match_targets[b]
            m = y.shape[0]
            for i in range(n):
                t = y[i, 0]
                for j in range(n):
                    idx = i * m + j
                    edges_mask[b, 0, idx] = 1
                    if j == t:
                        edges_target[b, 0, idx] = 1

        loss_edge = F.binary_cross_entropy(edges_pred, edges_target, edges_mask)
        loss_all = self.cfg.losses.weight_point * loss_point + self.cfg.losses.weight_edge * loss_edge

        tb_dict = {
            'loss_all': loss_all.item(),
            'loss_point': loss_point.item(),
            'loss_edge': loss_edge.item()
        }
        return loss_all, tb_dict, disp_dict


class DirectionalPointDetector(nn.modules.Module):
    """Detector for point with direction."""

    def __init__(self, cfg):
        super(DirectionalPointDetector, self).__init__()
        self.cfg = cfg
        input_channel_size = cfg.input_channels
        depth_factor = cfg.depth_factor
        output_channel_size = cfg.output_channels
        self.feature_extractor = YetAnotherDarknet(input_channel_size, depth_factor)

        layers = []
        layers += define_detector_block(16 * depth_factor)
        layers += define_detector_block(16 * depth_factor)
        layers += [nn.Conv2d(32 * depth_factor, output_channel_size,
                             kernel_size=1, stride=1, padding=0, bias=False)]
        self.predict = nn.Sequential(*layers)

        self.loss_func = nn.MSELoss().cuda()

    def forward(self, data_dict):
        img = data_dict['image']
        prediction = self.predict(self.feature_extractor(img))
        # 4 represents that there are 4 value: confidence, shape, offset_x,
        # offset_y, whose range is between [0, 1].
        point_pred, angle_pred = torch.split(prediction, 4, dim=1)
        point_pred = torch.sigmoid(point_pred)
        angle_pred = torch.tanh(angle_pred)
        points_pred = torch.cat((point_pred, angle_pred), dim=1)

        data_dict['points_pred'] = points_pred
        return data_dict

    def get_targets(self, data_dict):
        marks_gt_batch = data_dict['marks']
        npoints = data_dict['npoints']
        batch_size = marks_gt_batch.size()[0]
        targets = torch.zeros(batch_size, self.cfg.output_channels,
                              self.cfg.feature_map_size,
                              self.cfg.feature_map_size).cuda()

        mask = torch.zeros_like(targets)
        mask[:, 0].fill_(1.)

        for batch_idx, marks_gt in enumerate(marks_gt_batch):
            n = npoints[batch_idx].long()
            for marking_point in marks_gt[:n]:
                x, y = marking_point[:2]
                col = math.floor(x * self.cfg.feature_map_size)
                row = math.floor(y * self.cfg.feature_map_size)
                # Confidence Regression
                targets[batch_idx, 0, row, col] = 1.
                # Makring Point Shape Regression
                targets[batch_idx, 1, row, col] = marking_point[3]  # shape
                # Offset Regression
                targets[batch_idx, 2, row, col] = x * 16 - col
                targets[batch_idx, 3, row, col] = y * 16 - row
                # Direction Regression
                direction = marking_point[2]
                targets[batch_idx, 4, row, col] = math.cos(direction)
                targets[batch_idx, 5, row, col] = math.sin(direction)

                mask[batch_idx, 1:6, row, col].fill_(1.)
        return targets, mask

    def get_training_loss(self, data_dict):
        points_pred = data_dict['points_pred']
        targets, mask = self.get_targets(data_dict)

        disp_dict = {}

        loss_all = self.loss_func(points_pred * mask, targets * mask)

        tb_dict = {
            'loss_all': loss_all.item()
        }
        return loss_all, tb_dict, disp_dict

    def post_processing(self, data_dict):
        ret_dicts = {}
        pred_dicts = {}

        points_pred = data_dict['points_pred']

        points_pred_batch = []
        slots_pred = []
        for b, marks in enumerate(points_pred):
            points_pred = get_predicted_directional_points(marks, self.cfg.point_thresh, self.cfg.boundary_thresh)
            points_pred_batch.append(points_pred)

            slots_infer = self.inference_slots(points_pred)
            slots_tmp = []
            for (i, j) in slots_infer:
                score = min(points_pred[i][0], points_pred[j][0])
                x1, y1 = points_pred[i][1][:2]
                x2, y2 = points_pred[j][1][:2]
                tmp = (score, np.array([x1, y1, x2, y2]))
                slots_tmp.append(tmp)

            slots_pred.append(slots_tmp)

        pred_dicts['points_pred'] = points_pred_batch
        pred_dicts['slots_pred'] = slots_pred
        return pred_dicts, ret_dicts

    def inference_slots(self, marking_points):
        """Inference slots based on marking points."""
        VSLOT_MIN_DIST = 0.044771278151623496
        VSLOT_MAX_DIST = 0.1099427457599304
        HSLOT_MIN_DIST = 0.15057789144568634
        HSLOT_MAX_DIST = 0.44449496544202816
        SLOT_SUPPRESSION_DOT_PRODUCT_THRESH = 0.8

        num_detected = len(marking_points)
        slots = []
        for i in range(num_detected - 1):
            for j in range(i + 1, num_detected):
                point_i = marking_points[i]
                point_j = marking_points[j]
                # Step 1: length filtration.
                distance = calc_point_squre_dist(point_i[1], point_j[1])
                if not (VSLOT_MIN_DIST <= distance <= VSLOT_MAX_DIST
                        or HSLOT_MIN_DIST <= distance <= HSLOT_MAX_DIST):
                    continue
                # Step 2: pass through filtration.
                if pass_through_third_point(marking_points, i, j, SLOT_SUPPRESSION_DOT_PRODUCT_THRESH):
                    continue
                result = pair_marking_points(point_i, point_j)
                if result == 1:
                    slots.append((i, j))
                elif result == -1:
                    slots.append((j, i))
        return slots
