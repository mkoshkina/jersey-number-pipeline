from pathlib import Path
import sys
import os
import argparse


sushi = '/home/maria/hockeyMOT/SUSHI'
sys.path.append(str(sushi))  # add ROOT to PATH

_FASTREID_ROOT = sushi + '/fast-reid'
sys.path.append(str(_FASTREID_ROOT))
print(sys.path)

import numpy as np
import torch
from tqdm import tqdm
import cv2
from PIL import Image

from fastreid.engine import DefaultTrainer
from fastreid.data.build import build_transforms
from src.models.reid.fastreid_models import _get_cfg, _load_ckpt


# Using the fastreid model used by SUSHI MOT tracker and fine-tuned on SoccerNet tracking dataset
CONFIG_FILE = str(_FASTREID_ROOT+'/configs/Market1501/bagtricks_R50.yml')
MODEL_FILE = str(sushi+'/fastreid-models/model_weights/fastreid_soccernet_0e.pth')


def load_fastreid_model():
    cfg = _get_cfg(CONFIG_FILE, MODEL_FILE)
    model = DefaultTrainer.build_model(cfg)

    model = _load_ckpt(model, cfg)

    transforms = build_transforms(cfg, is_train=False)
    feature_embedding_model = model.eval()

    return feature_embedding_model, transforms

def generate_features(input_folder, output_folder):
    # load model
    feature_embedding_model, transforms = load_fastreid_model()
    use_cuda = True if torch.cuda.is_available() else False
    tracks = os.listdir(input_folder)

    for track in tqdm(tracks):
        features = []
        track_path = os.path.join(input_folder, track)
        images = os.listdir(track_path)
        output_file = os.path.join(output_folder, f"{track}_features.npy")
        for img_path in images:
            img = cv2.imread(os.path.join(track_path, img_path))
            input_img = Image.fromarray(img)
            input_img = torch.stack([transforms(input_img)])
            with torch.no_grad():
                global_feat = feature_embedding_model(input_img.cuda() if use_cuda else input_img)
            features.append(global_feat.cpu().numpy().reshape(-1,))

        np_feat = np.array(features)
        with open(output_file, 'wb') as f:
            np.save(f, np_feat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracklets_folder', help="Folder containing tracklet directories with images")
    parser.add_argument('--output_folder', help="Folder to store features in, one file per tracklet")
    args = parser.parse_args()

    #create if does not exist
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)

    generate_features(args.tracklets_folder, args.output_folder)



