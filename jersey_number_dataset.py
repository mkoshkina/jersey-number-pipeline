from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd
import json
from PIL import Image
from torchvision import transforms

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomGrayscale(),
        transforms.ColorJitter(brightness=.5, hue=.3),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ]),
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ]),
    'test': transforms.Compose([ # same as val
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ]),
}

class JerseyNumberDataset(Dataset):
    def __init__(self, annotations_file, img_dir, mode='train'):
        self.transform = data_transforms[mode]
        self.img_labels = pd.read_csv(annotations_file)
        unqiue_ids = np.unique(self.img_labels.iloc[:, 1].to_numpy())
        print(f"Datafile:{annotations_file}, number of labels:{len(self.img_labels)}, unique ids: {len(unqiue_ids)}")
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

class JerseyNumberMultitaskDataset(Dataset):
    def __init__(self, annotations_file, img_dir, mode='train'):
        self.transform = data_transforms[mode]
        self.img_labels = pd.read_csv(annotations_file)
        unqiue_ids = np.unique(self.img_labels.iloc[:, 1].to_numpy())
        print(f"Datafile:{annotations_file}, number of labels:{len(self.img_labels)}, unique ids: {len(unqiue_ids)}")
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    def get_digit_labels(self, label):
        if label < 10:
            return label, 10
        else:
            return label // 10, label % 10

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels.iloc[idx, 1]
        digit1, digit2 = self.get_digit_labels(label)
        if not (label> 0 and label < 100 and digit1 < 10 and digit1 > 0 and digit2 > -1 and digit2 < 11):
            print(label, digit1, digit2)
        if self.transform:
            image = self.transform(image)
        return image, label, digit1, digit2

class UnlabelledJerseyNumberLegibilityDataset(Dataset):
    def __init__(self, image_paths, mode='test'):
        self.transform = data_transforms[mode]
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image


class JerseyNumberLegibilityDataset(Dataset):
    def __init__(self, annotations_file, img_dir, mode='train', isBalanced=False):
        self.transform = data_transforms[mode]
        self.img_labels = pd.read_csv(annotations_file)
        if isBalanced:
            legible =self.img_labels[self.img_labels.iloc[:,1]==1]
            count_legible = len(legible)
            illegible = self.img_labels[self.img_labels.iloc[:,1]==0]
            illegible = illegible.sample(n=count_legible)
            self.img_labels = pd.concat([legible, illegible])
            print(f"Balanced dataset: legibles = {count_legible} all = {len(self.img_labels)}")
        else:
            legible = self.img_labels[self.img_labels.iloc[:, 1] == 1]
            count_legible = len(legible)
            print(f"As-is dataset: legibles = {count_legible} all = {len(self.img_labels)}")

        self.img_dir = img_dir


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)

        return image, label

