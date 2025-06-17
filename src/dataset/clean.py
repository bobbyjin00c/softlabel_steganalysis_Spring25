import os
import re

# 要处理的子目录
splits = ['train', 'val', 'test']
base_dir = 'dataset'  # 根据你的项目路径调整

# 匹配带后缀的文件：xxx_0.0.pgm ~ xxx_1.0.pgm
pattern = re.compile(r'.+_[0-9]\.[0-9]\.pgm$')

for split in splits:
    dir_path = os.path.join(base_dir, split)
    for fname in os.listdir(dir_path):
        if not fname.endswith('.pgm'):
            continue
        # 如果不符合带后缀格式，就删除
        if not pattern.match(fname):
            path = os.path.join(dir_path, fname)
            print(f"Removing original file: {path}")
            os.remove(path)

print("✅ Cleanup complete.")
