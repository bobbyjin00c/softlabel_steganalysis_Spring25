# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class SEBlock(nn.Module):
#     def __init__(self, channels, reduction=16):
#         super(SEBlock, self).__init__()
#         self.global_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channels, channels // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channels // reduction, channels, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.global_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y

# class StegoNet(nn.Module):
#     def __init__(self):
#         super(StegoNet, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             SEBlock(32),

#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             SEBlock(64),

#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             SEBlock(128),
#             nn.AdaptiveAvgPool2d((4, 4))
#         )

#         self.flatten = nn.Flatten()
#         self.fc = nn.Sequential(
#             nn.Linear(128 * 4 * 4, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(inplace=True)
#         )

#         # ✅ 优化后的分类头：更强的非线性表达
#         self.cls_head = nn.Sequential(
#             nn.Linear(256, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(512, 256),
#             nn.ReLU(inplace=True),
#             nn.Linear(256, 11)
#         )

#         self.reg_head = nn.Sequential(
#             nn.Linear(256, 1)
#         )

#         # ✅ 初始化分类头权重
#         for m in self.cls_head.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         x = self.features(x)
#         x = self.flatten(x)
#         x = self.fc(x)
#         cls_logits = self.cls_head(x)
#         reg_pred = torch.sigmoid(self.reg_head(x))  # 输出 ∈ [0, 1]
#         return cls_logits, reg_pred

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_lsb_hp_filter():
    base_kernels = torch.tensor([
        [[0, 0, 0], [0, 1, -1], [0, 0, 0]],     # 水平梯度
        [[0, 0, 0], [0, 1, 0], [0, -1, 0]],     # 垂直梯度
        [[0, 0, 0], [0, 1, 0], [0, 0, -1]],     # 对角线1
        [[0, 0, 0], [0, 1, 0], [-1, 0, 0]],     # 对角线2
        [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]]  # 拉普拉斯
    ], dtype=torch.float32)

    lsb_kernels = torch.tensor([
        [[1, -1, 0], [-1, 2, -1], [0, -1, 1]],    # LSB相关性检测
        [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],    # LSB中心增强
        [[-0.5, 1, -0.5], [1, -2, 1], [-0.5, 1, -0.5]]  # 位平面交叉检测
    ], dtype=torch.float32)

    all_kernels = torch.cat([base_kernels, lsb_kernels], dim=0)  # (8, 3, 3)
    return all_kernels.unsqueeze(1)  # (8, 1, 3, 3)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class StegoNet(nn.Module):
    def __init__(self):
        super(StegoNet, self).__init__()

        # 固定的 LSB 高通滤波层（不参与训练）
        self.lsb_filter = nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False)
        with torch.no_grad():
            self.lsb_filter.weight.copy_(get_lsb_hp_filter())
            self.lsb_filter.weight.requires_grad = False

        self.features = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SEBlock(32),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SEBlock(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SEBlock(128),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )

        # 分类头（11类 soft-label）
        self.cls_head = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 11)
        )

        # 回归头（sigmoid 输出 [0,1]）
        self.reg_head = nn.Sequential(
            nn.Linear(256, 1)
        )

        for m in self.cls_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.lsb_filter(x)         # (B, 8, H, W)，增强 LSB 特征
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        cls_logits = self.cls_head(x)
        reg_pred = torch.sigmoid(self.reg_head(x))  # 归一化输出
        return cls_logits, reg_pred
