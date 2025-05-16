from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union

class DetectionResult(BaseModel):
    """检测结果模型"""
    is_tampered: bool
    similarity: float
    threshold: float
    tampered_regions: List[Dict[str, int]]
    original_image: str
    encrypted_image: str 