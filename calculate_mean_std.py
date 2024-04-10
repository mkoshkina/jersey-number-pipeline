import os
import numpy as np
from PIL import Image

def rescale_image(img):
    return np.array(img) / 255.0

def get_image_list(root, partitions):
    images = []
    for part in partitions:
        img_dir = os.path.join(root, part, 'images')
        image_names = os.listdir(img_dir)
        for name in image_names:
            images.append(os.path.join(img_dir, name))
    return images

def get_image_list_soccer(root, partitions):
    images = []
    for part in partitions:
        img_dir = os.path.join(root, part, 'images')
        tracks = os.listdir(img_dir)
        for track in tracks:
            image_names = os.listdir(os.path.join(img_dir, track))
            for name in image_names:
                images.append(os.path.join(img_dir, track, name))
    return images

def calculate_mean_std(images):
    # Accumulators for sum and sum of squares
    sum_channels = np.zeros(3)
    sum_squares = np.zeros(3)
    total_pixels = 0

    for img_path in images:
        img = Image.open(img_path)
        img_rescaled = rescale_image(img)
        sum_channels += img_rescaled.sum(axis=(0, 1))
        sum_squares += (img_rescaled ** 2).sum(axis=(0, 1))
        total_pixels += img_rescaled.shape[0] * img_rescaled.shape[1]

    # Mean and standard deviation
    mean = sum_channels / total_pixels
    std = np.sqrt(sum_squares / total_pixels - mean ** 2)

    return mean, std

# Usage example
#root_path = '/home/maria/jersey-number-pipeline/data/Hockey/legibility_dataset'
#partitions = ['train', 'val', 'test']
root_path = '/home/maria/jersey-number-pipeline/data/SoccerNet'
partitions = ['train', 'test', 'challenge']
image_list = get_image_list_soccer(root_path, partitions)
print(len(image_list))
mean, std = calculate_mean_std(image_list)
print("Mean per channel:", mean)
print("Standard deviation per channel:", std)
