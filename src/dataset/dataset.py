import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms  # 添加transform库

class StegoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        # 默认transform：包含Resize、ToTensor和Normalize
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),  # 统一尺寸
            transforms.ToTensor(),          # 转为张量
            transforms.Normalize(mean=[0.5], std=[0.5])  # 添加均值标准差标准化
        ])
        self.image_paths = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith('.pgm')
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("L")  # 加载灰度图像
        
        # 使用transform处理图像（包含标准化）
        img = self.transform(img)
        
        # 从文件名解析嵌入率作为标签
        fname = os.path.join(img_path)
        try:
            # 尝试从文件名中提取嵌入率（如"train_00001_0.3.pgm"）
            base_name = os.path.basename(fname).replace('.pgm', '')
            rate = float(base_name.split('_')[-1])
        except (IndexError, ValueError):
            rate = 0.0  # 默认嵌入率为0

        label = torch.tensor([rate], dtype=torch.float32)
        return img, label