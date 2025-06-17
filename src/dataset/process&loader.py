import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def lsb_replace(img: np.ndarray, payload_rate: float) -> np.ndarray:
    flat = img.flatten()
    num_pixels = flat.shape[0]
    num_bits_to_embed = int(payload_rate * num_pixels)
    indices = np.random.choice(num_pixels, num_bits_to_embed, replace=False)
    secret_bits = np.random.randint(0, 2, num_bits_to_embed)
    for i, bit in zip(indices, secret_bits):
        flat[i] = (flat[i] & ~1) | bit
    return flat.reshape(img.shape)

def generate_stego_images(base_dir, embed_rates=[0.1 * i for i in range(1, 11)]):
    all_dirs = ['train', 'val', 'test']
    for folder in all_dirs:
        folder_path = os.path.join(base_dir, folder)
        for fname in os.listdir(folder_path):
            if not fname.endswith('.pgm'):
                continue  

            base_name = fname[:-4] 

            if len(base_name) > 3 and base_name[-3] == '_' and base_name[-1].isdigit():
                continue

            img_path = os.path.join(folder_path, fname)
            img = Image.open(img_path)
            img_np = np.array(img)

            orig_save_path = os.path.join(folder_path, f"{base_name}_0.0.pgm")
            if not os.path.exists(orig_save_path):
                Image.fromarray(img_np).save(orig_save_path)

            for rate in embed_rates:
                save_path = os.path.join(folder_path, f"{base_name}_{rate:.1f}.pgm")
                if not os.path.exists(save_path):
                    stego_np = lsb_replace(img_np.copy(), rate)
                    Image.fromarray(stego_np).save(save_path)

    print("隐写图像已全部生成完成。")

class StegoPGMDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.file_paths = []
        self.labels = []
        self.transform = transform

        for fname in os.listdir(directory):
            if fname.endswith('.pgm') and '_' in fname:
                fpath = os.path.join(directory, fname)
                self.file_paths.append(fpath)

                label_str = fname.split('_')[-1].replace('.pgm', '')
                try:
                    label = float(label_str)
                except:
                    label = 0.0
                self.labels.append(label)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = Image.open(self.file_paths[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, label

if __name__ == "__main__":
    base_dir = "dataset"  
    generate_stego_images(base_dir)

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = StegoPGMDataset(os.path.join(base_dir, "train"), transform=transform)
    print(f"训练集样本数: {len(train_dataset)}")
    img, label = train_dataset[0]
    print(f"示例图像尺寸: {img.shape}, 嵌入率标签: {label.item()}")
