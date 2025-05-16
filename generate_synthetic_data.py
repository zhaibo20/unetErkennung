"""
生成用于训练UNet模型的合成篡改数据集

此脚本生成原始图像、篡改图像和篡改掩码，用于训练UNet模型。
"""

import os
import cv2
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm

def create_random_shape(img_shape, min_size=30, max_size=100):
    """
    在图像上创建随机形状的篡改区域
    
    Args:
        img_shape: 图像形状 (高,宽)
        min_size: 最小篡改区域大小
        max_size: 最大篡改区域大小
        
    Returns:
        篡改掩码和篡改类型
    """
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # 篡改类型: 0=矩形, 1=圆形, 2=多边形
    tamper_type = random.randint(0, 2)
    
    # 随机位置
    x = random.randint(0, w - min_size)
    y = random.randint(0, h - min_size)
    
    # 随机大小
    size_w = random.randint(min_size, min(max_size, w - x))
    size_h = random.randint(min_size, min(max_size, h - y))
    
    if tamper_type == 0:  # 矩形
        mask[y:y+size_h, x:x+size_w] = 255
    elif tamper_type == 1:  # 圆形
        center = (x + size_w // 2, y + size_h // 2)
        radius = min(size_w, size_h) // 2
        cv2.circle(mask, center, radius, 255, -1)
    else:  # 多边形
        num_points = random.randint(3, 8)
        points = []
        center_x = x + size_w // 2
        center_y = y + size_h // 2
        
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            r = random.uniform(min_size / 2, min(size_w, size_h) / 2)
            px = int(center_x + r * np.cos(angle))
            py = int(center_y + r * np.sin(angle))
            points.append([px, py])
            
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
    
    return mask, tamper_type

def apply_tamper(img, mask, tamper_type):
    """
    应用篡改效果到图像
    
    Args:
        img: 原始图像
        mask: 篡改掩码
        tamper_type: 篡改类型
        
    Returns:
        篡改后的图像
    """
    tampered = img.copy()
    
    # 获取篡改区域
    x, y, w, h = cv2.boundingRect(mask)
    
    if tamper_type == 0:  # 复制-粘贴篡改
        # 随机选择源区域
        src_x = random.randint(0, img.shape[1] - w)
        src_y = random.randint(0, img.shape[0] - h)
        
        # 复制区域
        src_region = img[src_y:src_y+h, src_x:src_x+w].copy()
        
        # 创建过渡区域
        kernel = np.ones((5, 5), np.uint8)
        expanded_mask = cv2.dilate(mask[y:y+h, x:x+w], kernel, iterations=2)
        blurred_mask = cv2.GaussianBlur(expanded_mask, (5, 5), 0)
        
        # 应用篡改
        for c in range(3):
            tampered[y:y+h, x:x+w, c] = (src_region[:, :, c] * (blurred_mask / 255) + 
                                       tampered[y:y+h, x:x+w, c] * (1 - blurred_mask / 255))
            
    elif tamper_type == 1:  # 内容移除和填充
        # 使用内容感知填充
        tampered = cv2.inpaint(tampered, mask, 3, cv2.INPAINT_TELEA)
        
    else:  # 几何变换
        # 对篡改区域应用几何变换
        sub_img = tampered[y:y+h, x:x+w].copy()
        
        # 随机变换类型
        transform_type = random.randint(0, 2)
        
        if transform_type == 0:  # 旋转
            angle = random.uniform(-30, 30)
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            sub_img = cv2.warpAffine(sub_img, M, (w, h))
        elif transform_type == 1:  # 透视变换
            pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            pts2 = np.float32([
                [random.randint(0, w//4), random.randint(0, h//4)],
                [random.randint(3*w//4, w), random.randint(0, h//4)],
                [random.randint(0, w//4), random.randint(3*h//4, h)],
                [random.randint(3*w//4, w), random.randint(3*h//4, h)]
            ])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            sub_img = cv2.warpPerspective(sub_img, M, (w, h))
        else:  # 缩放
            scale = random.uniform(0.8, 1.2)
            new_w = int(w * scale)
            new_h = int(h * scale)
            sub_img = cv2.resize(sub_img, (new_w, new_h))
            sub_img = cv2.resize(sub_img, (w, h))
            
        # 应用到原图
        sub_mask = mask[y:y+h, x:x+w]
        for c in range(3):
            tampered[y:y+h, x:x+w, c] = sub_mask / 255 * sub_img[:, :, c] + (1 - sub_mask / 255) * tampered[y:y+h, x:x+w, c]
            
    return tampered

def generate_dataset(num_images, output_dir, img_size=(512, 512)):
    """
    生成合成篡改数据集
    
    Args:
        num_images: 生成图像数量
        output_dir: 输出目录
        img_size: 图像大小
    """
    # 创建输出目录
    output_dir = Path(output_dir)
    original_dir = output_dir / "original"
    tampered_dir = output_dir / "tampered"
    masks_dir = output_dir / "masks"
    
    for d in [original_dir, tampered_dir, masks_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    print(f"生成 {num_images} 对图像...")
    
    for i in tqdm(range(num_images)):
        # 创建随机颜色的图像或随机噪声
        if random.random() < 0.5:
            # 随机颜色图像
            img = np.ones((img_size[0], img_size[1], 3), dtype=np.uint8) * random.randint(0, 255)
            # 添加随机形状
            num_shapes = random.randint(5, 20)
            for _ in range(num_shapes):
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                shape_type = random.randint(0, 2)
                if shape_type == 0:  # 矩形
                    pt1 = (random.randint(0, img_size[1]), random.randint(0, img_size[0]))
                    pt2 = (random.randint(pt1[0], img_size[1]), random.randint(pt1[1], img_size[0]))
                    cv2.rectangle(img, pt1, pt2, color, -1)
                elif shape_type == 1:  # 圆形
                    center = (random.randint(0, img_size[1]), random.randint(0, img_size[0]))
                    radius = random.randint(10, 100)
                    cv2.circle(img, center, radius, color, -1)
                else:  # 多边形
                    num_points = random.randint(3, 8)
                    points = []
                    center_x = random.randint(100, img_size[1]-100)
                    center_y = random.randint(100, img_size[0]-100)
                    for j in range(num_points):
                        angle = 2 * np.pi * j / num_points
                        r = random.randint(20, 100)
                        px = int(center_x + r * np.cos(angle))
                        py = int(center_y + r * np.sin(angle))
                        points.append([px, py])
                    points = np.array(points, dtype=np.int32)
                    cv2.fillPoly(img, [points], color)
        else:
            # 随机噪声图像
            img = np.random.randint(0, 255, (img_size[0], img_size[1], 3), dtype=np.uint8)
            # 应用高斯模糊使噪声更真实
            img = cv2.GaussianBlur(img, (5, 5), 0)
        
        # 创建篡改掩码
        num_tampers = random.randint(1, 3)  # 每张图像1-3个篡改区域
        mask = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)
        
        for _ in range(num_tampers):
            sub_mask, tamper_type = create_random_shape(img.shape)
            mask = cv2.bitwise_or(mask, sub_mask)
        
        # 应用篡改
        tampered = apply_tamper(img, mask, random.randint(0, 2))
        
        # 保存图像
        filename = f"{i:05d}"
        cv2.imwrite(str(original_dir / f"{filename}.png"), img)
        cv2.imwrite(str(tampered_dir / f"{filename}.png"), tampered)
        cv2.imwrite(str(masks_dir / f"{filename}.png"), mask)
        
    print(f"数据集生成完成! 保存在 {output_dir}")

if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    random.seed(42)
    
    # 配置参数
    num_images = 10  # 生成的图像对数量
    output_dir = "data/tampering_dataset"  # 输出目录
    img_size = (256, 256)  # 图像大小
    
    generate_dataset(num_images, output_dir, img_size) 