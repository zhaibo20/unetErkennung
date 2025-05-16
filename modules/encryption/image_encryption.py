import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from cryptography.fernet import Fernet
import base64
import hashlib

from config.settings import ENCRYPTION_KEY, ENCRYPTED_DIR

class ImageEncryption:
    def __init__(self):
        # 使用配置的密钥或生成新密钥
        key = ENCRYPTION_KEY.encode()
        if len(key) < 32:
            key = hashlib.sha256(key).digest()
        self.key = base64.urlsafe_b64encode(key)
        self.cipher_suite = Fernet(self.key)

    def encrypt_image(self, image_path: Path) -> Optional[Path]:
        """加密图像文件"""
        try:
            # 读取图像
            img = cv2.imread(str(image_path))
            if img is None:
                return None

            # 将图像转换为字节
            _, img_encoded = cv2.imencode('.jpg', img)
            img_bytes = img_encoded.tobytes()

            # 加密数据
            encrypted_data = self.cipher_suite.encrypt(img_bytes)

            # 保存加密后的图像
            encrypted_path = ENCRYPTED_DIR / f"encrypted_{image_path.name}"
            with open(encrypted_path, 'wb') as f:
                f.write(encrypted_data)

            return encrypted_path
        except Exception as e:
            print(f"加密图像失败: {str(e)}")
            return None

    def decrypt_image(self, encrypted_path: Path) -> Optional[np.ndarray]:
        """解密图像文件"""
        try:
            # 读取加密数据
            with open(encrypted_path, 'rb') as f:
                encrypted_data = f.read()

            # 解密数据
            decrypted_data = self.cipher_suite.decrypt(encrypted_data)

            # 将字节转换回图像
            nparr = np.frombuffer(decrypted_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            return img
        except Exception as e:
            print(f"解密图像失败: {str(e)}")
            return None

    def verify_image(self, original_path: Path, encrypted_path: Path) -> bool:
        """验证图像是否被篡改"""
        try:
            # 读取原始图像
            original_img = cv2.imread(str(original_path))
            if original_img is None:
                return False

            # 解密图像
            decrypted_img = self.decrypt_image(encrypted_path)
            if decrypted_img is None:
                return False

            # 比较图像
            return np.array_equal(original_img, decrypted_img)
        except Exception as e:
            print(f"验证图像失败: {str(e)}")
            return False 