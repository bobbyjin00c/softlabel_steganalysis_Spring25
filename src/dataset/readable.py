import os
from PIL import Image
from tqdm import tqdm

input_base = 'dataset'
output_base = 'dataset_png'

splits = ['train', 'val', 'test']
for split in splits:
    os.makedirs(os.path.join(output_base, split), exist_ok=True)

def convert_pgm_to_png():
    for split in splits:
        input_dir = os.path.join(input_base, split)
        output_dir = os.path.join(output_base, split)
        
        print(f"Processing {split}...")
        for filename in tqdm(os.listdir(input_dir)):
            if filename.endswith('.pgm'):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, filename.replace('.pgm', '.png'))

                try:
                    with Image.open(input_path) as img:
                        img = img.convert('L') 
                        img.save(output_path)
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")

if __name__ == '__main__':
    convert_pgm_to_png()
    print("All PGM files converted to PNG.")
