import numpy as np
from scipy.stats import norm
from tqdm import tqdm
import numpy as np
import json
import os
import argparse

def get_main_subject(image_folder, feature_folder, threshold = 3.5, rounds = 3):
    tracks = os.listdir(image_folder)

    results = {}
    for r in range(rounds):
        results[r] = {x: [] for x in tracks}

    for tr in tqdm(tracks):
        images = os.listdir(os.path.join(image_folder, tr))
        features_path = os.path.join(feature_folder, f"{tr}_features.npy")
        #print(features_path)
        with open(features_path, 'rb') as f:
            features = np.load(f)
        if len(images) <= 2:
            results[tr] = images
            continue

        cleaned_data = features
        for r in range(rounds):
            # Fit a Gaussian distribution to the data
            mu = np.mean(cleaned_data, axis=0)

            euclidean_distance = np.linalg.norm(features - mu, axis = 1)

            mean_euclidean_distance = np.mean(euclidean_distance)
            std = np.std(euclidean_distance)
            th = threshold * std

            # Remove outliers from the data
            cleaned_data = features[(euclidean_distance - mean_euclidean_distance) <= threshold]
            cleaned_data_indexes = np.where((euclidean_distance - mean_euclidean_distance)<= threshold)[0]


            for i in cleaned_data_indexes:
                results[r][tr].append(images[i])

    for r in range(rounds):
        result_file_name = f"main_subject_gauss_th={threshold}_r={r + 1}.json"
        with open(os.path.join(feature_folder, result_file_name), "w") as outfile:
            json.dump(results[r], outfile)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracklets_folder', help="Folder containing tracklet directories with images")
    parser.add_argument('--output_folder', help="Folder to store features in, one file per tracklet")
    parser.add_argument('--threshold', type=float, default=3.5,  required=False, help="Threshold for outlier removal per round, used to compute distance threshold*std")
    parser.add_argument('--rounds',  type=int, default=3,  required=False, help="Number of iteration for outlier removal")
    args = parser.parse_args()

    get_main_subject(args.tracklets_folder, args.output_folder, threshold=args.threshold, rounds=args.rounds)
