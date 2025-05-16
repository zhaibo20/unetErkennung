import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

from config.settings import CAMERA_INDEX, FRAME_RATE, UPLOAD_DIR

class CameraCapture:
    def __init__(self, camera_index: int = CAMERA_INDEX):
        self.camera_index = camera_index
        self.camera = None
        self.is_running = False

    def start(self) -> bool:
        """启动摄像头"""
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            self.camera.set(cv2.CAP_PROP_FPS, FRAME_RATE)
            self.is_running = True
            return True
        except Exception as e:
            print(f"启动摄像头失败: {str(e)}")
            return False

    def stop(self):
        """停止摄像头"""
        if self.camera is not None:
            self.camera.release()
        self.is_running = False

    def capture_frame(self) -> Optional[np.ndarray]:
        """捕获一帧图像"""
        if not self.is_running or self.camera is None:
            return None
        
        ret, frame = self.camera.read()
        if not ret:
            return None
        return frame

    def save_frame(self, frame: np.ndarray) -> Optional[Path]:
        """保存图像到文件"""
        if frame is None:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.jpg"
        filepath = UPLOAD_DIR / filename

        try:
            cv2.imwrite(str(filepath), frame)
            return filepath
        except Exception as e:
            print(f"保存图像失败: {str(e)}")
            return None

    def capture_and_save(self) -> Optional[Path]:
        """捕获并保存一帧图像"""
        frame = self.capture_frame()
        if frame is None:
            return None
        return self.save_frame(frame)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop() 