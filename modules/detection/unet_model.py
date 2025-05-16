import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class DoubleConv(nn.Module):
    """双层卷积块（Conv2d -> BN -> ReLU）x 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """下采样路径: 最大池化 + 双层卷积"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """上采样路径: 上采样 + 拼接 + 双层卷积"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # 上采样层
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 计算填充大小
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # 拼接特征图
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """输出卷积层"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """UNet模型实现"""
    def __init__(self, n_channels=3, n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 下采样路径
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # 上采样路径
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return torch.sigmoid(logits)  # 使用sigmoid激活输出0-1之间的值

def preprocess_images_for_unet(img1, img2, target_size=(256, 256)):
    """
    预处理图像用于UNet模型输入
    
    Args:
        img1: 原始图像 (OpenCV格式，BGR)
        img2: 目标图像 (OpenCV格式，BGR)
        target_size: 模型输入大小
        
    Returns:
        tensor: 包含两张图像的张量，形状为 [1, 6, H, W]
    """
    # 调整图像大小
    img1 = cv2.resize(img1, target_size)
    img2 = cv2.resize(img2, target_size)
    
    # 转换为RGB通道
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # 归一化
    img1 = img1.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0
    
    # 将两张图像拼接在通道维度
    combined = np.concatenate([img1, img2], axis=2)
    
    # 转换为PyTorch张量并添加batch维度
    tensor = torch.from_numpy(combined.transpose(2, 0, 1)).unsqueeze(0)
    
    return tensor

def postprocess_unet_output(output_tensor, original_shape):
    """
    处理UNet模型输出，转换为篡改区域
    
    Args:
        output_tensor: 模型输出张量，形状为 [1, 1, H, W]
        original_shape: 原始图像形状
        
    Returns:
        np.ndarray: 篡改区域掩码
    """
    # 获取预测掩码
    pred_mask = output_tensor.squeeze().cpu().detach().numpy()
    
    # 二值化掩码
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    
    # 调整大小为原始图像大小
    pred_mask = cv2.resize(pred_mask, (original_shape[1], original_shape[0]))
    
    return pred_mask 