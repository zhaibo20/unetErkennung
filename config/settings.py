import os
from pathlib import Path

# 基础配置
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
ENCRYPTED_DIR = BASE_DIR / "static" / "encrypted"
LOG_DIR = BASE_DIR / "logs"

# 创建必要的目录
for directory in [UPLOAD_DIR, ENCRYPTED_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# 加密配置
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "your-secret-key-here")  # 在生产环境中应该使用环境变量

# 摄像头配置
CAMERA_INDEX = 0  # 默认摄像头索引
FRAME_RATE = 30   # 帧率

# 检测配置
DETECTION_THRESHOLD = 0.95  # 图像相似度阈值
ALERT_ENABLED = True       # 是否启用报警

# API配置
API_HOST = "127.0.0.1"
API_PORT = 8000
DEBUG = True

# UNet模型配置
USE_UNET = True  # 是否使用UNet模型
UNET_MODEL_PATH = BASE_DIR / "modules" / "detection" / "unet_model.pth"
UNET_INPUT_SIZE = (256, 256)  # 模型输入图像大小
UNET_MASK_THRESHOLD = 0.5  # 预测掩码的二值化阈值 