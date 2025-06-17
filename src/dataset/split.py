import os
import random
import shutil
from tqdm import tqdm

def split_and_rename_datasets(source_dir1, source_dir2,
                             train_count1=7000, train_count2=7000,
                             val_count_per_source=500):
    """
    随机分类并标准化命名数据集，直接输出到train/test/val文件夹
    
    参数:
        source_dir1: BossBase文件夹路径
        source_dir2: BOWS2文件夹路径
        train_count1: 从BossBase选取的训练集数量
        train_count2: 从BOWS2选取的训练集数量
        val_count_per_source: 从每个源选取的验证集数量
    """
    # 创建独立的输出文件夹（与脚本同级）
    os.makedirs('train', exist_ok=True)
    os.makedirs('test', exist_ok=True)
    os.makedirs('val', exist_ok=True)

    # 获取文件列表并验证
    bossbase_files = [f for f in os.listdir(source_dir1) if f.lower().endswith('.pgm')]
    bows2_files = [f for f in os.listdir(source_dir2) if f.lower().endswith('.pgm')]
    
    assert len(bossbase_files) >= train_count1 + val_count_per_source, "BossBase文件不足"
    assert len(bows2_files) >= train_count2 + val_count_per_source, "BOWS2文件不足"

    # 随机打乱文件列表（确保可复现性可添加random.seed）
    random.shuffle(bossbase_files)
    random.shuffle(bows2_files)

    # 划分BossBase文件
    bossbase_train = bossbase_files[:train_count1]
    bossbase_val = bossbase_files[train_count1:train_count1 + val_count_per_source]
    bossbase_test = bossbase_files[train_count1 + val_count_per_source:]

    # 划分BOWS2文件
    bows2_train = bows2_files[:train_count2]
    bows2_val = bows2_files[train_count2:train_count2 + val_count_per_source]
    bows2_test = bows2_files[train_count2 + val_count_per_source:]

    # 文件计数器初始化
    counters = {'train': 1, 'val': 1, 'test': 1}

    def copy_files(files, set_type, source_dir):
        """通用复制和重命名函数"""
        for file in tqdm(files, desc=f"处理 {set_type} 文件"):
            dst_name = f"{set_type}_{counters[set_type]:05d}.pgm"
            shutil.copy(
                os.path.join(source_dir, file),
                os.path.join(set_type, dst_name)
            )
            counters[set_type] += 1

    # 处理BossBase文件
    copy_files(bossbase_train, 'train', source_dir1)
    copy_files(bossbase_val, 'val', source_dir1)
    copy_files(bossbase_test, 'test', source_dir1)

    # 处理BOWS2文件
    copy_files(bows2_train, 'train', source_dir2)
    copy_files(bows2_val, 'val', source_dir2)
    copy_files(bows2_test, 'test', source_dir2)

    # 验证数量
    print("\n✅ 数据集划分完成：")
    print(f"训练集: {counters['train']-1} 文件 (BossBase: {len(bossbase_train)}, BOWS2: {len(bows2_train)})")
    print(f"验证集: {counters['val']-1} 文件 (BossBase: {len(bossbase_val)}, BOWS2: {len(bows2_val)})")
    print(f"测试集: {counters['test']-1} 文件 (BossBase: {len(bossbase_test)}, BOWS2: {len(bows2_test)})")

if __name__ == "__main__":
    # 配置路径（请修改为实际路径）
    bossbase_dir = "dataset\BossBase"
    bows2_dir = "dataset\BOWS2"
    
    # 执行划分（输出到同级目录的train/test/val文件夹）
    split_and_rename_datasets(
        source_dir1=bossbase_dir,
        source_dir2=bows2_dir,
        train_count1=7000,
        train_count2=7000,
        val_count_per_source=500
    )