import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime
import logging
import base64
from cryptography.fernet import Fernet
import hashlib
from skimage.metrics import structural_similarity as ssim
import os
import torch

from config.settings import DETECTION_THRESHOLD, ALERT_ENABLED, ENCRYPTION_KEY
from config.settings import USE_UNET, UNET_MODEL_PATH, UNET_INPUT_SIZE, UNET_MASK_THRESHOLD
from modules.detection.unet_model import UNet, preprocess_images_for_unet, postprocess_unet_output

logger = logging.getLogger(__name__)

class TamperDetection:
    def __init__(self, DETECTION_THRESHOLD=0.95):
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
        self.DETECTION_THRESHOLD = DETECTION_THRESHOLD
        # 初始化加密密钥
        key = ENCRYPTION_KEY.encode()
        if len(key) < 32:
            key = hashlib.sha256(key).digest()
        self.key = base64.urlsafe_b64encode(key)
        self.cipher_suite = Fernet(self.key)
        
        # 初始化UNet模型
        self.use_unet = USE_UNET
        if self.use_unet:
            try:
                logger.info("正在初始化UNet模型...")
                # 初始化UNet模型 (输入为两张图像的6个通道)
                self.unet_model = UNet(n_channels=6, n_classes=1)
                
                # 检查是否存在预训练模型
                if UNET_MODEL_PATH.exists():
                    logger.info(f"加载UNet模型权重: {UNET_MODEL_PATH}")
                    # 加载模型权重
                    self.unet_model.load_state_dict(torch.load(UNET_MODEL_PATH))
                    # 设置为评估模式
                    self.unet_model.eval()
                else:
                    logger.warning(f"未找到UNet模型权重文件: {UNET_MODEL_PATH}")
                    logger.warning("将退回到传统图像处理方法")
                    self.use_unet = False
            except Exception as e:
                logger.error(f"初始化UNet模型失败: {str(e)}")
                logger.warning("将退回到传统图像处理方法")
                self.use_unet = False

    def calculate_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """计算两张图片的相似度"""
        try:
            # 检查图像是否完全相同
            if np.array_equal(img1, img2):
                return 1.0
                
            # 确保两张图片大小相同
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

            # 转换为灰度图
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # 计算直方图
            hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

            # 归一化直方图
            cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

            # 计算直方图相似度
            hist_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

            # 计算结构相似性指数 (SSIM)
            ssim_value = ssim(gray1, gray2, data_range=255)

            # 综合两种相似度
            final_similarity = (hist_similarity + ssim_value) / 2

            # 确保相似度在0-1之间
            final_similarity = max(0.0, min(1.0, final_similarity))

            return final_similarity
        except Exception as e:
            logger.error(f"计算相似度失败: {str(e)}")
            return 0.0

    def _decrypt_image(self, encrypted_path: Path) -> Optional[np.ndarray]:
        """解密图像文件"""
        try:
            # 读取加密数据
            with open(encrypted_path, 'rb') as f:
                encrypted_data = f.read()

            # 解密数据
            decrypted_data = self.cipher_suite.decrypt(encrypted_data)

            # 将解密后的数据转换为图像
            nparr = np.frombuffer(decrypted_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            logger.error(f"解密图像失败: {str(e)}")
            return None

    def detect_tampering(self, original_path, encrypted_path):
        """
        检测两个图像之间的篡改情况
        
        Args:
            original_path: 原始图像路径
            encrypted_path: 待检测图像路径(可能是加密图像或普通图像)
            
        Returns:
            dict: 包含检测结果的字典
        """
        # 检查文件是否存在
        if not os.path.exists(original_path):
            raise ValueError(f"原始图像不存在: {original_path}")
            
        if not os.path.exists(encrypted_path):
            raise ValueError(f"待检测图像不存在: {encrypted_path}")
            
        try:
            # 读取原始图像
            original_img = cv2.imread(str(original_path))
            if original_img is None:
                raise ValueError(f"无法读取原始图像: {original_path}")
            
            # 尝试解密待检测图像
            encrypted_img = None
            
            # 先尝试作为普通图像读取
            encrypted_img = cv2.imread(str(encrypted_path))
            
            # 如果普通读取失败，尝试解密
            if encrypted_img is None:
                try:
                    encrypted_img = self._decrypt_image(encrypted_path)
                except Exception as e:
                    logger.warning(f"图像解密失败，尝试作为普通图像处理: {str(e)}")
            
            # 如果两种方式都无法读取图像，则报错
            if encrypted_img is None:
                raise ValueError(f"无法读取待检测图像: {encrypted_path}")
            
            # 确保两个图像尺寸相同
            if original_img.shape != encrypted_img.shape:
                # 将encrypted_img调整为与original_img相同的尺寸
                encrypted_img = cv2.resize(encrypted_img, (original_img.shape[1], original_img.shape[0]))
            
            # 计算相似度
            similarity = self.calculate_similarity(original_img, encrypted_img)
            
            # 检测篡改区域
            tampered_regions = self.detect_tampered_regions(original_img, encrypted_img)
            
            # 判断图像是否被篡改（根据是否存在篡改区域判断）
            is_tampered = len(tampered_regions) > 0
            
            return {
                "is_tampered": is_tampered,
                "similarity": similarity,
                "tampered_regions": tampered_regions,
                "threshold": self.DETECTION_THRESHOLD  # 仍然返回阈值，以便前端显示
            }
            
        except Exception as e:
            logger.error(f"检测篡改时出错: {str(e)}")
            raise

    def detect_tampered_regions(self, original_img: np.ndarray, target_img: np.ndarray) -> list:
        """检测图像中被篡改的区域
        
        Args:
            original_img: 原始图像
            target_img: 目标图像（可能被篡改）
            
        Returns:
            list: 篡改区域的矩形坐标 [x, y, width, height]
        """
        try:
            # 确保两张图片大小相同
            if original_img.shape != target_img.shape:
                target_img = cv2.resize(target_img, (original_img.shape[1], original_img.shape[0]))
            
            # 如果启用了UNet模型，优先使用UNet进行检测
            if self.use_unet:
                try:
                    logger.info("使用UNet模型检测篡改区域")
                    
                    # 预处理图像
                    input_tensor = preprocess_images_for_unet(original_img, target_img, UNET_INPUT_SIZE)
                    
                    # 使用UNet预测篡改区域
                    with torch.no_grad():
                        output = self.unet_model(input_tensor)
                    
                    # 后处理结果，获取篡改掩码
                    mask = postprocess_unet_output(output, original_img.shape)
                    
                    # 查找轮廓
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    logger.info(f"UNet检测到 {len(contours)} 个潜在篡改区域")
                    
                    # 筛选面积大于阈值的区域
                    min_area = 20  # 最小区域面积
                    tampered_regions = []
                    
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area > min_area:
                            x, y, w, h = cv2.boundingRect(contour)
                            tampered_regions.append({
                                "x": int(x),
                                "y": int(y),
                                "width": int(w),
                                "height": int(h)
                            })
                            logger.info(f"UNet检测到篡改区域: x={x}, y={y}, w={w}, h={h}, 面积={area}")
                    
                    # 如果UNet检测到篡改区域，直接返回结果
                    if tampered_regions:
                        return tampered_regions
                    
                    # 如果UNet没有检测到篡改区域，使用传统方法作为备选
                    logger.info("UNet未检测到篡改区域，使用传统方法作为备选")
                except Exception as e:
                    logger.error(f"UNet检测失败: {str(e)}")
                    logger.info("使用传统方法检测篡改区域")
            
            # 传统方法：使用图像处理技术检测篡改区域
            # 转为灰度图
            gray1 = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
            
            # 提高图像对比度
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray1 = clahe.apply(gray1)
            gray2 = clahe.apply(gray2)
            
            # 计算图像差异
            diff = cv2.absdiff(gray1, gray2)
            
            # 设置阈值，获取差异明显的区域
            _, threshold = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)  # 降低阈值增加敏感度
            
            # 进行形态学操作，合并相近的区域
            kernel = np.ones((3, 3), np.uint8)  # 使用更小的内核
            dilated = cv2.dilate(threshold, kernel, iterations=1)
            
            # 可选：执行闭运算，填充空洞
            closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # 查找轮廓
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            logger.info(f"传统方法检测到 {len(contours)} 个潜在篡改区域")
            
            # 筛选出面积大于一定阈值的区域，以排除噪点
            min_area = 20  # 降低最小面积阈值
            tampered_regions = []
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    # 扩大边界框以包含更多区域
                    x = max(0, x - 5)
                    y = max(0, y - 5)
                    w = min(gray1.shape[1] - x, w + 10)
                    h = min(gray1.shape[0] - y, h + 10)
                    
                    tampered_regions.append({
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h)
                    })
                    logger.info(f"篡改区域 #{len(tampered_regions)}: x={x}, y={y}, w={w}, h={h}, 面积={area}")
            
            # 合并重叠区域
            i = 0
            while i < len(tampered_regions):
                j = i + 1
                r1 = tampered_regions[i]
                merged = False
                
                while j < len(tampered_regions):
                    r2 = tampered_regions[j]
                    
                    # 检查是否有交集
                    if (r1["x"] < r2["x"] + r2["width"] and
                        r1["x"] + r1["width"] > r2["x"] and
                        r1["y"] < r2["y"] + r2["height"] and
                        r1["y"] + r1["height"] > r2["y"]):
                        
                        # 合并区域
                        x = min(r1["x"], r2["x"])
                        y = min(r1["y"], r2["y"])
                        w = max(r1["x"] + r1["width"], r2["x"] + r2["width"]) - x
                        h = max(r1["y"] + r1["height"], r2["y"] + r2["height"]) - y
                        
                        r1.update({"x": x, "y": y, "width": w, "height": h})
                        tampered_regions.pop(j)
                        merged = True
                    else:
                        j += 1
                
                if not merged:
                    i += 1
            
            logger.info(f"合并后共有 {len(tampered_regions)} 个篡改区域")
            
            return tampered_regions
            
        except Exception as e:
            logger.error(f"检测篡改区域失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def _trigger_alert(self, original_path: Path, current_path: Path, similarity: float):
        """触发篡改警报"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert_message = (
            f"[{timestamp}] 检测到图像篡改!\n"
            f"原始图像: {original_path}\n"
            f"当前图像: {current_path}\n"
            f"相似度: {similarity:.2f}\n"
            f"阈值: {self.DETECTION_THRESHOLD}"
        )
        print(alert_message)
        # 这里可以添加更多的警报方式，如发送邮件、推送通知等 