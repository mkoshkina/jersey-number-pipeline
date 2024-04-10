from PIL import Image
from pathlib import Path
import torch
import torchvision

# if you change the seed, make sure that the randomly-applied transforms
# properly show that the image can be both transformed and *not* transformed!
torch.manual_seed(0)

orig_img = Image.open(Path('/media/storage/jersey_ids/SNLegibilityVal03/val/images/1_66.jpg'))
grayscale = torchvision.transforms.Grayscale()
transformed_img = grayscale(orig_img)

# with open('/home/maria/sample_transforms/grayscale.jpg', 'w') as f:
#     transformed_img.save(f)

sharpness = torchvision.transforms.RandomAdjustSharpness(3, p=1)
transformed_img = sharpness(orig_img)
with open('/home/maria/sample_transforms/sharpness.jpg', 'w') as f:
    transformed_img.save(f)

contrast = torchvision.transforms.RandomAutocontrast(p=1)
transformed_img = contrast(orig_img)
with open('/home/maria/sample_transforms/contrast.jpg', 'w') as f:
    transformed_img.save(f)

blur = torchvision.transforms.GaussianBlur(kernel_size=(5, 9), sigma=(.8, 5.))
transformed_img = blur(orig_img)
with open('/home/maria/sample_transforms/dblur.jpg', 'w') as f:
    transformed_img.save(f)

color = torchvision.transforms.ColorJitter(brightness=.8, contrast=0.8, saturation=0.5, hue=0.1)
transformed_img = color(orig_img)
with open('/home/maria/sample_transforms/color2.jpg', 'w') as f:
    transformed_img.save(f)