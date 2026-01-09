import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import pprint

# 自动将项目根目录加入 Python 路径
ROOT = Path(__file__).resolve().parent.parent  # 根据你的脚本位置调整 .parent 次数
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from psdet.utils import dist
from psdet.utils.config import get_config
from psdet.utils.common import get_logger, set_random_seed

from psdet.models.builder import build_model
from psdet.datasets.builder import build_dataloader

from eval_utils import eval_utils


def eval_single_model(model, test_loader, cfg, ckpt_file, logger, dist_test):
    logger.info('Evaluting {:s}'.format(str(ckpt_file)))
    # load checkpoint
    model.load_params_from_file(filename=ckpt_file, logger=logger, to_cpu=dist_test)
    model.cuda()

    # start evaluation   
    eval_utils.eval_point_detection(cfg, model, test_loader, logger, dist_test=dist_test,
                                    result_dir=cfg.output_dir, save_to_file=cfg.get('save_to_file', True))

def enhance_white_lines(img):
    """
    增强图像中的白色/浅色线条（适用于 PS2.0 BEV 图像）
    输入: BGR uint8 图像 (H, W, 3)
    输出: 增强后的 BGR uint8 图像
    """
    # 1. 转到 HSV 空间，提取高亮度、低饱和度区域（即白色）
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 白色范围：低饱和度 + 高明度
    mask_white = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))

    # 2. 提取边缘（Canny）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # 3. 合并：白色区域 + 边缘 → 强化标线
    combined_mask = cv2.bitwise_or(mask_white, edges)

    # 4. 对原图进行对比度拉伸（可选 CLAHE）
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 5. 将强化的标线区域叠加回图像（提高亮度）
    # 创建一个亮白色的 overlay
    overlay = np.full_like(img, 255)
    # 只在 combined_mask 区域叠加
    alpha = 0.3  # 叠加强度
    img_enhanced = img_clahe.copy()
    img_enhanced[combined_mask > 0] = cv2.addWeighted(
        img_clahe[combined_mask > 0], 1 - alpha,
        overlay[combined_mask > 0], alpha,
        0
    ).astype(np.uint8)

    return img_enhanced

def main():
    cfg = get_config()
    logger = get_logger(cfg.log_dir, cfg.tag)
    # log to file
    logger.info(pprint.pformat(cfg))

    if cfg.launcher == 'none':
        dist_test = False
    else:
        logger.info('Start distributed testing ...')
        cfg.batch_size, cfg.local_rank = dist.init_dist_pytorch(
            cfg.batch_size, cfg.local_rank, backend='nccl'
        )
        cfg.data.val.batch_size = cfg.batch_size
        dist_test = True

    if dist_test:
        total_gpus = dist.get_world_size()
        logger.info('total_batch_size: %d' % (total_gpus * cfg.batch_size))

    test_set, test_loader, sampler = build_dataloader(
        cfg.data.val, dist=dist_test, training=False, logger=logger)

    model = build_model(cfg.model)
    # logger.info(model)

    with torch.no_grad():
        if cfg.eval_all:
            ckpt_files = list(cfg.model_dir.glob('*.pth'))
            ckpt_files = sorted(ckpt_files, key=os.path.getmtime, reverse=True)
            for ckpt_file in ckpt_files:
                eval_single_model(model, test_loader, cfg, ckpt_file=ckpt_file, logger=logger, dist_test=dist_test)
        else:
            eval_single_model(model, test_loader, cfg, ckpt_file=cfg.ckpt, logger=logger, dist_test=dist_test)


if __name__ == '__main__':
    main()
