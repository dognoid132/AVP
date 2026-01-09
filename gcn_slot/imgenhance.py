import cv2
import numpy as np


import cv2
import numpy as np

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
    edge_color = (0, 0, 255)  # 红色
    final_result[edges_canny > 0] = edge_color

    return final_result



# 读取图像
img_path = r'E:\gcn-parking-slot-main\images\img.jpg'  # 替换为你的图像路径
img_path = r'E:\gcn-parking-slot-main\images\021.jpg'  # 替换为你的图像路径
original_img = cv2.imread(img_path)

# 检查图像是否加载成功
if original_img is None:
    print("图像加载失败，请检查文件路径")
else:
    # 应用增强函数
    enhanced_img = enhance_lane_lines(original_img.copy())

    # 显示原图和增强后的图像对比
    combined_img = np.hstack((original_img, enhanced_img))
    cv2.imshow('Before and After', combined_img)
    cv2.waitKey(0)  # 等待按键关闭窗口
    cv2.destroyAllWindows()  # 关闭所有OpenCV创建的窗口
