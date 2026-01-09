import sys

import cv2
import time
import torch
import pprint
import numpy as np
from pathlib import Path

# 自动将项目根目录加入 Python 路径
ROOT = Path(__file__).resolve().parent.parent  # 根据你的脚本位置调整 .parent 次数
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from psdet.utils.config import get_config
from psdet.utils.common import get_logger
from psdet.models.builder import build_model

sys.argv = [
    'tools/demo.py',  # 脚本名（可任意，但建议保留）
    '-c', r'E:\gcn-parking-slot-main\config\ps_gat.yaml',  # 配置文件路径
    '-m', r'E:\gcn-parking-slot-main\retrain_weight\checkpoint_epoch_210.pth'  # 模型权重路径
]


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

def draw_parking_slot(image, pred_dicts):
    slots_pred = pred_dicts['slots_pred']

    width = 512
    height = 512
    VSLOT_MIN_DIST = 0.044771278151623496
    VSLOT_MAX_DIST = 0.1099427457599304
    HSLOT_MIN_DIST = 0.15057789144568634
    HSLOT_MAX_DIST = 0.44449496544202816

    SHORT_SEPARATOR_LENGTH = 0.199519231
    LONG_SEPARATOR_LENGTH = 0.46875
    junctions = []
    for j in range(len(slots_pred[0])):
        position = slots_pred[0][j][1]
        p0_x = width * position[0] - 0.5
        p0_y = height * position[1] - 0.5
        p1_x = width * position[2] - 0.5
        p1_y = height * position[3] - 0.5
        vec = np.array([p1_x - p0_x, p1_y - p0_y])
        vec = vec / np.linalg.norm(vec)
        distance = (position[0] - position[2]) ** 2 + (position[1] - position[3]) ** 2

        if VSLOT_MIN_DIST <= distance <= VSLOT_MAX_DIST:
            separating_length = LONG_SEPARATOR_LENGTH
        else:
            separating_length = SHORT_SEPARATOR_LENGTH

        p2_x = p0_x + height * separating_length * vec[1]
        p2_y = p0_y - width * separating_length * vec[0]
        p3_x = p1_x + height * separating_length * vec[1]
        p3_y = p1_y - width * separating_length * vec[0]
        p0_x = int(round(p0_x))
        p0_y = int(round(p0_y))
        p1_x = int(round(p1_x))
        p1_y = int(round(p1_y))
        p2_x = int(round(p2_x))
        p2_y = int(round(p2_y))
        p3_x = int(round(p3_x))
        p3_y = int(round(p3_y))
        cv2.line(image, (p0_x, p0_y), (p1_x, p1_y), (255, 0, 0), 2)
        cv2.line(image, (p0_x, p0_y), (p2_x, p2_y), (255, 0, 0), 2)
        cv2.line(image, (p1_x, p1_y), (p3_x, p3_y), (255, 0, 0), 2)

        # cv2.circle(image, (p0_x, p0_y), 3,  (0, 0, 255), 4)
        junctions.append((p0_x, p0_y))
        junctions.append((p1_x, p1_y))
    for junction in junctions:
        cv2.circle(image, junction, 3, (0, 0, 255), 4)

    return image


def main():
    cfg = get_config()
    logger = get_logger(cfg.log_dir, cfg.tag)
    logger.info(pprint.pformat(cfg))

    model = build_model(cfg.model)
    logger.info(model)
    ### 2026/1/4为了windows适配修改
    # image_dir = Path(cfg.data_root) / 'testing' / 'outdoor-normal daylight'
    image_dir = Path(r'E:\gcn-parking-slot-main\datasets\parking_slot\testing\outdoor-normal daylight')
    display = True

    # load checkpoint
    model.load_params_from_file(filename=cfg.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    if display:
        car = cv2.imread(r'E:\gcn-parking-slot-main\images\car.png')
        car = cv2.resize(car, (512, 512))

    with torch.no_grad():

        for img_path in image_dir.glob('*.jpg'):
            img_name = img_path.stem

            data_dict = {}
            image = cv2.imread(str(img_path))
            image0 = cv2.resize(image, (512, 512))
            image0 = enhance_white_lines(image0)
            image = image0 / 255.

            data_dict['image'] = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0).cuda()

            start_time = time.time()
            pred_dicts, ret_dict = model(data_dict)
            sec_per_example = (time.time() - start_time)
            print('Info speed: %.4f second per example.' % sec_per_example)

            if display:
                image = draw_parking_slot(image0, pred_dicts)
                image[145:365, 210:300] = 0
                image += car
                cv2.imshow('image', image.astype(np.uint8))
                cv2.waitKey(50)

                save_dir = Path(cfg.output_dir) / 'predictions'
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / ('%s.jpg' % img_name)
                cv2.imwrite(str(save_path), image)
    if display:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
