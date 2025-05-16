import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse
from pathlib import Path
import shutil
import base64
from typing import List, Optional
import json
from datetime import datetime
import cv2
import logging
import os

from modules.capture.camera import CameraCapture
from modules.encryption.image_encryption import ImageEncryption
from modules.detection.tamper_detection import TamperDetection
from modules.storage.image_storage import ImageStorage
from config.settings import UPLOAD_DIR, ENCRYPTED_DIR, API_HOST, API_PORT, LOG_DIR, BASE_DIR
from schemas import DetectionResult

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="图像采集与安全检测系统")

# 确保静态文件目录存在
static_dir = Path("static")
uploads_dir = static_dir / "uploads"
encrypted_dir = static_dir / "encrypted"

for directory in [static_dir, uploads_dir, encrypted_dir]:
    directory.mkdir(parents=True, exist_ok=True)

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
app.mount("/uploads", StaticFiles(directory=str(uploads_dir)), name="uploads")
app.mount("/encrypted", StaticFiles(directory=str(encrypted_dir)), name="encrypted")

# 添加favicon.ico处理
@app.get('/favicon.ico')
async def favicon():
    return RedirectResponse(url="/static/favicon.ico")

# 初始化各个模块
camera = CameraCapture()
encryption = ImageEncryption()
detection = TamperDetection(DETECTION_THRESHOLD=0.95)
storage = ImageStorage()

@app.get("/")
async def root():
    """根路径重定向到index.html"""
    return RedirectResponse(url="/static/index.html")

@app.post("/capture")
async def capture_image(request: Request):
    """使用摄像头采集图像"""
    try:
        # 获取前端发送的图片数据
        data = await request.json()
        image_data = data.get('image')
        if not image_data:
            raise HTTPException(status_code=400, detail="未收到图片数据")

        # 解码base64图片数据
        image_bytes = base64.b64decode(image_data.split(',')[1])
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = UPLOAD_DIR / f"capture_{timestamp}.jpg"
        with open(image_path, 'wb') as f:
            f.write(image_bytes)
        logger.info(f"保存采集的图像: {image_path}")

        # 加密图像
        encrypted_path = encryption.encrypt_image(image_path)
        if not encrypted_path:
            raise HTTPException(status_code=500, detail="图像加密失败")
        logger.info(f"加密图像: {encrypted_path}")

        # 保存记录
        if not storage.add_image(image_path, encrypted_path):
            raise HTTPException(status_code=500, detail="保存记录失败")
        logger.info("成功保存图像记录")

        return {
            "message": "图像采集成功",
            "image_id": image_path.stem,
            "original_path": str(image_path),
            "encrypted_path": str(encrypted_path)
        }
    except Exception as e:
        logger.error(f"图像采集失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_image(original: UploadFile = File(...), detect: UploadFile = File(...)):
    """上传原始图像和待检测图像"""
    try:
        # 保存上传的文件
        original_path = UPLOAD_DIR / original.filename
        detect_path = UPLOAD_DIR / detect.filename
        
        with original_path.open("wb") as buffer:
            shutil.copyfileobj(original.file, buffer)
        with detect_path.open("wb") as buffer:
            shutil.copyfileobj(detect.file, buffer)
        logger.info(f"保存上传的图像: 原图 {original_path.relative_to(Path.cwd())}, 检测图 {detect_path.relative_to(Path.cwd())}")

        # 进行篡改检测
        try:
            result = detection.detect_tampering(
                original_path,
                detect_path
            )
            is_tampered = result["is_tampered"]
            similarity = result["similarity"]
            logger.info(f"检测结果 - 是否篡改: {is_tampered}, 相似度: {similarity}")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"检测过程出错: {str(e)}")

        # 加密原始图像
        encrypted_path = encryption.encrypt_image(original_path)
        if not encrypted_path:
            raise HTTPException(status_code=500, detail="图像加密失败")
        logger.info(f"加密图像: {encrypted_path.relative_to(Path.cwd())}")

        # 保存记录
        if not storage.add_image(original_path, encrypted_path):
            raise HTTPException(status_code=500, detail="保存记录失败")
        logger.info("成功保存图像记录")

        return {
            "message": "图像上传成功",
            "image_id": original_path.stem,
            "original_path": str(original_path.relative_to(Path.cwd())),
            "detect_path": str(detect_path.relative_to(Path.cwd())),
            "is_tampered": is_tampered,
            "similarity": similarity,
            "threshold": result["threshold"]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"上传图像失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"上传图像失败: {str(e)}")

@app.get("/images")
async def list_images(page: int = 1, page_size: int = 6, search: Optional[str] = None):
    """获取图像列表，支持分页和搜索"""
    try:
        # 使用更新的存储方法获取分页数据
        result = storage.list_images(page=page, page_size=page_size, search_query=search)
        logger.info(f"获取图像列表成功，页码: {page}, 每页数量: {page_size}, 共 {result['pagination']['total_images']} 张")
        return result
    except Exception as e:
        logger.error(f"获取图像列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/images/{image_id}")
async def get_image_info(image_id: str):
    """获取指定图像信息"""
    try:
        info = storage.get_image_info(image_id)
        if not info:
            raise HTTPException(status_code=404, detail="图像不存在")
        return info
    except Exception as e:
        logger.error(f"获取图像信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/images/{image_id}")
async def delete_image(image_id: str):
    """删除指定图像"""
    try:
        if not storage.delete_image(image_id):
            raise HTTPException(status_code=404, detail="图像不存在")
        logger.info(f"成功删除图像: {image_id}")
        return {"message": "图像删除成功"}
    except Exception as e:
        logger.error(f"删除图像失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/{image_id}", response_model=DetectionResult)
async def detect_image(image_id: str):
    """检测图像是否被篡改"""
    try:
        # 获取图像信息
        image_info = storage.get_image_info(image_id)
        if not image_info:
            raise HTTPException(status_code=404, detail="图像不存在")
            
        # 获取原始图像和加密图像路径
        original_path = Path(image_info['original_path'])
        encrypted_path = Path(image_info['encrypted_path'])
        
        # 检测图像是否被篡改
        result = detection.detect_tampering(
            original_path, 
            encrypted_path
        )
        
        # 读取原始图像并转为base64编码
        with open(original_path, "rb") as img_file:
            original_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            
        # 解密加密图像并转为base64
        decrypted_img = detection._decrypt_image(encrypted_path)
        if decrypted_img is not None:
            # 将解密后的图像转换为base64
            _, buffer = cv2.imencode('.jpg', decrypted_img)
            encrypted_base64 = base64.b64encode(buffer).decode('utf-8')
        else:
            # 如果解密失败，使用原始加密文件
            with open(encrypted_path, "rb") as img_file:
                encrypted_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            
        # 返回检测结果
        return {
            "is_tampered": result["is_tampered"],
            "similarity": result["similarity"],
            "threshold": result["threshold"],
            "tampered_regions": result["tampered_regions"],
            "original_image": original_base64,
            "encrypted_image": encrypted_base64
        }
    except Exception as e:
        logger.error(f"检测图像失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"检测图像失败: {str(e)}")

@app.post("/compare")
async def compare_images(original: UploadFile = File(...), compare: UploadFile = File(...)):
    """对比两张图片是否被篡改"""
    try:
        # 保存上传的文件
        original_path = UPLOAD_DIR / original.filename
        compare_path = UPLOAD_DIR / compare.filename
        
        with original_path.open("wb") as buffer:
            shutil.copyfileobj(original.file, buffer)
        with compare_path.open("wb") as buffer:
            shutil.copyfileobj(compare.file, buffer)
        logger.info(f"保存对比图像: {original_path}, {compare_path}")

        # 检测篡改
        try:
            result = detection.detect_tampering(
                original_path,
                compare_path
            )
            is_tampered = result["is_tampered"]
            similarity = result["similarity"]
            logger.info(f"对比检测结果 - 是否篡改: {is_tampered}, 相似度: {similarity}")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"检测过程出错: {str(e)}")

        # 清理临时文件
        original_path.unlink()
        compare_path.unlink()
        logger.info("清理临时文件")

        return {
            "is_tampered": is_tampered,
            "similarity": similarity,
            "threshold": result["threshold"]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"对比检测失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"对比检测失败: {str(e)}")

@app.post("/upload_multiple")
async def upload_multiple_images(files: List[UploadFile] = File(...)):
    """上传多个图像文件"""
    try:
        uploaded_count = 0
        uploaded_files = []
        
        for file in files:
            # 跳过非图像文件
            if not file.content_type.startswith('image/'):
                continue
                
            # 保存上传的文件
            file_path = UPLOAD_DIR / file.filename
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(f"保存上传的图像: {file_path}")

            # 加密图像
            encrypted_path = encryption.encrypt_image(file_path)
            if not encrypted_path:
                logger.warning(f"图像加密失败: {file_path}")
                continue
            logger.info(f"加密图像: {encrypted_path}")

            # 保存记录
            if not storage.add_image(file_path, encrypted_path):
                logger.warning(f"保存记录失败: {file_path}")
                continue
            
            uploaded_count += 1
            uploaded_files.append({
                "original_path": str(file_path),
                "encrypted_path": str(encrypted_path),
                "filename": file.filename
            })

        logger.info(f"成功上传 {uploaded_count} 个文件")
        return {
            "message": f"成功上传 {uploaded_count} 个文件",
            "uploaded_count": uploaded_count,
            "uploaded_files": uploaded_files
        }
    except Exception as e:
        logger.error(f"多图像上传失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect_uploaded")
async def detect_uploaded_images(original: UploadFile = File(...), detect: UploadFile = File(...)):
    """检测上传的两张图片是否被篡改"""
    try:
        # 保存上传的文件
        original_path = UPLOAD_DIR / f"temp_{original.filename}"
        detect_path = UPLOAD_DIR / f"temp_{detect.filename}"
        
        with original_path.open("wb") as buffer:
            shutil.copyfileobj(original.file, buffer)
        with detect_path.open("wb") as buffer:
            shutil.copyfileobj(detect.file, buffer)
        logger.info(f"保存临时检测图像: 原图 {original_path}, 检测图 {detect_path}")

        # 转换图像为base64以便在前端显示
        original_image = ""
        detect_image = ""
        
        try:
            with open(str(original_path), 'rb') as f:
                original_image = base64.b64encode(f.read()).decode('utf-8')
            logger.info("已读取原始图像数据")
        except Exception as e:
            logger.warning(f"读取原始图像失败: {str(e)}")
            
        try:
            with open(str(detect_path), 'rb') as f:
                detect_image = base64.b64encode(f.read()).decode('utf-8')
            logger.info("已读取检测图像数据")
        except Exception as e:
            logger.warning(f"读取检测图像失败: {str(e)}")

        # 检测篡改
        try:
            # 检查文件类型是否支持
            for path in [original_path, detect_path]:
                try:
                    img = cv2.imread(str(path))
                    if img is None:
                        raise HTTPException(status_code=400, detail=f"不支持的图像格式或图像损坏: {path.name}")
                except Exception:
                    raise HTTPException(status_code=400, detail=f"读取图像失败: {path.name}")
            
            result = detection.detect_tampering(
                original_path,
                detect_path
            )
            is_tampered = result.get("is_tampered", False)
            similarity = result.get("similarity", 0.0)
            tampered_regions = result.get("tampered_regions", [])
            logger.info(f"检测结果 - 是否篡改: {is_tampered}, 相似度: {similarity}, 篡改区域数量: {len(tampered_regions)}")
            print(f"检测到 {len(tampered_regions)} 个篡改区域: {tampered_regions}")
        except ValueError as e:
            logger.error(f"检测值错误: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"检测过程出错: {str(e)}")
            raise HTTPException(status_code=500, detail=f"检测过程出错: {str(e)}")

        # 清理临时文件
        try:
            if original_path.exists():
                original_path.unlink()
            if detect_path.exists():
                detect_path.unlink()
            logger.info("清理临时文件完成")
        except Exception as e:
            logger.warning(f"清理临时文件失败: {str(e)}")

        return {
            "is_tampered": is_tampered,
            "similarity": similarity,
            "tampered_regions": tampered_regions,
            "original_image": original_image,
            "encrypted_image": detect_image
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"检测上传图像失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info"
    ) 