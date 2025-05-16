import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

from config.settings import UPLOAD_DIR, ENCRYPTED_DIR, LOG_DIR, BASE_DIR

logger = logging.getLogger(__name__)

class ImageStorage:
    def __init__(self):
        self.metadata_file = LOG_DIR / "image_metadata.json"
        self._load_metadata()
        logger.info(f"初始化图像存储，元数据文件路径: {self.metadata_file}")

    def _to_relative_path(self, path_str: str) -> str:
        """将路径转换为相对于项目根目录的路径"""
        try:
            path = Path(path_str)
            if not path.is_absolute():
                # 如果已经是相对路径，直接返回
                return str(path)
            try:
                # 尝试转换为相对于项目根目录的路径
                return str(path.relative_to(BASE_DIR))
            except ValueError:
                # 如果不是项目根目录的子目录，尝试转换为相对于当前工作目录的路径
                return str(path.relative_to(Path.cwd()))
        except Exception as e:
            logger.warning(f"路径转换失败: {str(e)}")
            return path_str

    def _to_absolute_path(self, path_str: str) -> str:
        """将相对路径转换为绝对路径"""
        path = Path(path_str)
        if path.is_absolute():
            return str(path)
        return str(BASE_DIR / path)

    def _load_metadata(self):
        """加载图像元数据"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                    # 确保所有路径都是相对路径
                    for image_id, info in self.metadata.items():
                        if "original_path" in info:
                            info["original_path"] = self._to_relative_path(info["original_path"])
                        if "encrypted_path" in info:
                            info["encrypted_path"] = self._to_relative_path(info["encrypted_path"])
                logger.info(f"成功加载元数据，包含 {len(self.metadata)} 条记录")
            else:
                self.metadata = {}
                logger.info("元数据文件不存在，创建新的元数据")
        except Exception as e:
            logger.error(f"加载元数据失败: {str(e)}")
            self.metadata = {}

    def _save_metadata(self):
        """保存图像元数据"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"成功保存元数据，包含 {len(self.metadata)} 条记录")
        except Exception as e:
            logger.error(f"保存元数据失败: {str(e)}")

    def add_image(self, original_path: Path, encrypted_path: Path) -> bool:
        """添加图像记录"""
        try:
            image_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            logger.info(f"添加新图像记录: {image_id}")
            
            # 转换为相对路径
            rel_original_path = self._to_relative_path(str(original_path))
            rel_encrypted_path = self._to_relative_path(str(encrypted_path))
            
            logger.info(f"原始图像路径: {rel_original_path}")
            logger.info(f"加密图像路径: {rel_encrypted_path}")
            
            if not Path(self._to_absolute_path(rel_original_path)).exists():
                logger.error(f"原始图像文件不存在: {original_path}")
                return False
                
            if not Path(self._to_absolute_path(rel_encrypted_path)).exists():
                logger.error(f"加密图像文件不存在: {encrypted_path}")
                return False

            self.metadata[image_id] = {
                "original_path": rel_original_path,
                "encrypted_path": rel_encrypted_path,
                "timestamp": datetime.now().isoformat(),
                "status": "active"
            }
            self._save_metadata()
            logger.info(f"成功添加图像记录: {image_id}")
            return True
        except Exception as e:
            logger.error(f"添加图像记录失败: {str(e)}")
            return False

    def get_image_info(self, image_id: str) -> Optional[Dict]:
        """获取图像信息"""
        info = self.metadata.get(image_id)
        if info:
            # 转换为绝对路径返回
            info = info.copy()
            info["original_path"] = self._to_absolute_path(info["original_path"])
            info["encrypted_path"] = self._to_absolute_path(info["encrypted_path"])
            logger.info(f"获取图像信息成功: {image_id}")
        else:
            logger.warning(f"图像不存在: {image_id}")
        return info

    def list_images(self, page=1, page_size=6, search_query=None) -> dict:
        """
        列出图像，支持分页和搜索功能
        
        Args:
            page: 当前页码，从1开始
            page_size: 每页显示的图像数量
            search_query: 搜索关键词（搜索图像ID或路径）
            
        Returns:
            dict: 包含图像数据和分页信息的字典
        """
        # 过滤图像数据
        if search_query and search_query.strip():
            search_query = search_query.lower()
            filtered_images = [
                {"id": image_id, **info}
                for image_id, info in self.metadata.items()
                if (search_query in image_id.lower() or
                    search_query in info.get("original_path", "").lower() or
                    search_query in info.get("timestamp", "").lower())
            ]
        else:
            filtered_images = [
                {"id": image_id, **info}
                for image_id, info in self.metadata.items()
            ]
        
        # 对图像按时间戳降序排序（最新的在前面）
        filtered_images.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # 计算分页信息
        total_images = len(filtered_images)
        total_pages = (total_images + page_size - 1) // page_size  # 向上取整
        
        # 确保页码在有效范围内
        page = max(1, min(page, total_pages)) if total_pages > 0 else 1
        
        # 计算切片范围
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_images)
        
        # 获取当前页的图像
        current_page_images = filtered_images[start_idx:end_idx] if total_images > 0 else []
        
        logger.info(f"列出图像，页码: {page}/{total_pages}, 每页: {page_size}, 搜索: {search_query}, 找到: {total_images}张")
        
        return {
            "images": current_page_images,
            "pagination": {
                "current_page": page,
                "total_pages": total_pages,
                "page_size": page_size,
                "total_images": total_images
            }
        }

    def delete_image(self, image_id: str) -> bool:
        """删除图像记录"""
        try:
            if image_id in self.metadata:
                info = self.metadata[image_id]
                logger.info(f"删除图像: {image_id}")
                # 删除文件
                for path in [info["original_path"], info["encrypted_path"]]:
                    if os.path.exists(path):
                        os.remove(path)
                        logger.info(f"删除文件: {path}")
                    else:
                        logger.warning(f"文件不存在: {path}")
                # 删除记录
                del self.metadata[image_id]
                self._save_metadata()
                logger.info(f"成功删除图像记录: {image_id}")
                return True
            logger.warning(f"图像不存在: {image_id}")
            return False
        except Exception as e:
            logger.error(f"删除图像失败: {str(e)}")
            return False

    def update_image_status(self, image_id: str, status: str) -> bool:
        """更新图像状态"""
        try:
            if image_id in self.metadata:
                self.metadata[image_id]["status"] = status
                self._save_metadata()
                logger.info(f"更新图像状态成功: {image_id} -> {status}")
                return True
            logger.warning(f"图像不存在: {image_id}")
            return False
        except Exception as e:
            logger.error(f"更新图像状态失败: {str(e)}")
            return False

    def get_image_by_status(self, status: str) -> List[Dict]:
        """获取指定状态的图像"""
        images = [
            {"id": image_id, **info}
            for image_id, info in self.metadata.items()
            if info["status"] == status
        ]
        logger.info(f"获取状态为 {status} 的图像，共 {len(images)} 张")
        return images 