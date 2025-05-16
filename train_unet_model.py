"""
训练UNet模型用于图像篡改检测

此脚本用于训练UNet模型，将训练好的模型保存为 modules/detection/unet_model.pth
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import random
from tqdm import tqdm

from modules.detection.unet_model import UNet

# 训练数据集类
class TamperDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        数据集初始化
        
        Args:
            data_dir: 数据集目录，包含原始图像和篡改图像的子目录
            transform: 数据增强变换
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # 获取所有原始图像路径
        self.original_images = list((self.data_dir / "original").glob("*.jpg"))
        self.original_images.extend(list((self.data_dir / "original").glob("*.png")))
        
        # 获取所有篡改图像路径
        self.tampered_images = list((self.data_dir / "tampered").glob("*.jpg"))
        self.tampered_images.extend(list((self.data_dir / "tampered").glob("*.png")))
        
        # 获取所有篡改掩码路径
        self.masks = list((self.data_dir / "masks").glob("*.png"))
        
        # 确保数据集长度一致
        assert len(self.original_images) == len(self.tampered_images) == len(self.masks), \
            "原始图像、篡改图像和掩码的数量必须相同"
            
    def __len__(self):
        return len(self.original_images)
        
    def __getitem__(self, idx):
        # 读取图像
        original_img = cv2.imread(str(self.original_images[idx]))
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        tampered_img = cv2.imread(str(self.tampered_images[idx]))
        tampered_img = cv2.cvtColor(tampered_img, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(str(self.masks[idx]), cv2.IMREAD_GRAYSCALE)
        
        # 调整图像大小
        original_img = cv2.resize(original_img, (256, 256))
        tampered_img = cv2.resize(tampered_img, (256, 256))
        mask = cv2.resize(mask, (256, 256))
        
        # 归一化
        original_img = original_img.astype(np.float32) / 255.0
        tampered_img = tampered_img.astype(np.float32) / 255.0
        mask = (mask > 128).astype(np.float32)
        
        # 拼接原始图像和篡改图像
        combined = np.concatenate([original_img, tampered_img], axis=2)
        
        # 转换为Tensor
        combined = torch.from_numpy(combined.transpose(2, 0, 1))
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        # 数据增强
        if self.transform:
            combined = self.transform(combined)
            mask = self.transform(mask)
            
        return combined, mask

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    """
    训练UNet模型
    
    Args:
        model: UNet模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 训练轮数
        device: 训练设备
        
    Returns:
        训练好的模型
    """
    model.to(device)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for inputs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            inputs = inputs.to(device)
            masks = masks.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                inputs = inputs.to(device)
                masks = masks.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item() * inputs.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "modules/detection/unet_model.pth")
            print(f"Model saved with Val Loss: {val_loss:.4f}")
            
    return model

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 数据集路径
    data_dir = "data/tampering_dataset"
    
    # 确保目录存在
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, "original"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "tampered"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "masks"), exist_ok=True)
    
    # 检查数据集
    print("请确保数据集目录结构为:")
    print(f"{data_dir}/original/  # 原始图像")
    print(f"{data_dir}/tampered/  # 篡改图像")
    print(f"{data_dir}/masks/     # 篡改掩码 (二值图像)")
    
    # 检查训练数据是否存在
    original_images = list(Path(data_dir).glob("original/*.jpg"))
    original_images.extend(list(Path(data_dir).glob("original/*.png")))
    
    if len(original_images) == 0:
        print("错误: 未找到训练数据。请先准备训练数据集!")
        print("您可以使用合成数据生成脚本: python generate_synthetic_data.py")
        return
        
    print(f"找到 {len(original_images)} 张训练图像")
    
    # 创建数据集和数据加载器
    # 数据增强
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])
    
    # 准备数据集
    dataset = TamperDataset(data_dir, transform=None)
    
    # 划分训练集和验证集
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
    
    print(f"训练集: {train_size} 张图像")
    print(f"验证集: {val_size} 张图像")
    
    # 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model = UNet(n_channels=6, n_classes=1)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    print("开始训练...")
    trained_model = train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        num_epochs=50,
        device=device
    )
    
    print("训练完成!")
    print("模型已保存到: modules/detection/unet_model.pth")
    
if __name__ == "__main__":
    main() 