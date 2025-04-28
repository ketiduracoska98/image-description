import os
import random
import shutil

source_dir = 'val2017'
output_dir = 'dataset'

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

all_images = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

selected_images = random.sample(all_images, 20)

for image in selected_images:
    source_path = os.path.join(source_dir, image)
    destination_path = os.path.join(output_dir, image)

    shutil.copy(source_path, destination_path)

print(f"20 imagini au fost copiate în directorul '{output_dir}'")
