import torch
import os
class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    train_data_dir = os.path.join(BASE_DIR, 'dataset', 'train')
    val_data_dir   = os.path.join(BASE_DIR, 'dataset', 'val')
    test_data_dir  = os.path.join(BASE_DIR, 'dataset', 'test')

    USE_REGRESSION = True  # 是否启用回归

    # 混合精度相关
    USE_AMP = True
    GRAD_CLIP = 1.0  # 梯度裁剪阈值

    LOSS_ALPHA = 0.7

    batch_size = 128
    # batch_size = 64
    # lr = 1e-3
    lr = 3e-4
    num_epochs = 30
    patience = 3

    checkpoint_path = './checkpoints/stegonet_cls.pt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

