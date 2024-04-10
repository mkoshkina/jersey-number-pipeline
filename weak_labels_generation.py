from tqdm import tqdm
import legibility_classifier as lc
import shutil
import argparse
import os
import random
import json
import configuration as cfg
from pathlib import Path
from natsort import natsorted
import helpers
import numpy as np
import pandas as pd

illegible_tracks_file = "/media/storage/jersey_ids/SNLegibilityFeb5/illegible_train.json"
legible_tracks_file = "/media/storage/jersey_ids/SNLegibilityFeb5/legible_train.json"

train_imgs = "images"
train_gt_path = "train_gt.json"
legibility_model = cfg.dataset["Hockey"]["legibility_model"]
SKIP = 5

def get_legibility(data_path, labels, return_legible, th):
    legible_images = []
    illegible_images = []

    for directory in tqdm(labels.keys()):
        if labels[directory] == -1:
            continue
        track_dir = os.path.join(data_path, directory)
        images = natsorted(os.listdir(track_dir))
        images_full_path = [os.path.join(track_dir, x) for x in images]
        track_results = lc.run(images_full_path, legibility_model, threshold=th)
        for i, r in enumerate(track_results):
            if r == 0:
                illegible_images.append(images_full_path[i])
            else:
                legible_images.append(images_full_path[i])

    if return_legible:
        return legible_images
    else:
        return illegible_images


# run classifier on all labelled
# compile a list of legible
# copy all legible files to legible folder
# compile the list of illegible mixing labelled and unlabelled, make the list the same size as illegible
# copy illegible to a folder

# create combined gt and split into train and val
def generate_legibility_dataset(src, dst, use_backup=True):
    illegible_tracks = []
    legible_tracks = []
    illegible_images, legible_images = None, None
    train_dir = os.path.join(src, "train")
    with open(os.path.join(train_dir, train_gt_path), 'r') as f:
        labels = json.load(f)
    for track in labels.keys():
        if labels[track] == -1:
            illegible_tracks.append(track)
        else:
            legible_tracks.append(track)

    img_path = os.path.join(train_dir, train_imgs)
    legible_backup_path = os.path.join(dst, "legible.json")
    if use_backup:
        legible_backup_path = legible_tracks_file
        # if available load intermidiate results
        with open(legible_backup_path, 'r') as f:
            legible_images = json.load(f)
            #legible_images = backup["legible"]
    else:
        legible_images = get_legibility(img_path, labels, True, 0.85)
        # backup intermidiate results
        with open(legible_backup_path, 'w') as f:
            json.dump({'legible': legible_images}, f)

    illegible_backup_path = os.path.join(dst, "illegible.json")
    if use_backup:
        illegible_backup_path = illegible_tracks_file
        # if available load intermidiate results
        with open(illegible_backup_path, 'r') as f:
            illegible_images = json.load(f)
            #illegible_images = backup["illegible"]
    else:
        illegible_tracks = get_legibility(img_path, labels, False, 0.3)
        # backup intermidiate results
        with open(illegible_backup_path, 'w') as f:
            json.dump({'illegible': illegible_images}, f)

    legible = []
    for key in legible_images.keys():
        for i, p in enumerate(legible_images[key]):
            if i % SKIP == 0:
                legible.append(p)

    print(f"Saving {len(legible)} legible ")
    illegible_images = []
    # sample a number of illegible
    for directory in illegible_tracks:
        track_dir = os.path.join(img_path, directory)
        images = os.listdir(track_dir)
        images_full_path = [os.path.join(track_dir, x) for x in images]

        for i, p in enumerate(images_full_path):
            illegible_images.append(p)

    if len(illegible_images) < len(legible):
        illegible = illegible_images
    else:
        illegible = random.sample(illegible_images, len(legible))


    # generate one gt and then split into train and val
    duplicates_count = 0
    for il in illegible:
        for l in legible:
            if os.path.basename(il) == os.path.basename(l):
                print(f"Duplicate found {il}")
                duplicates_count += 1
                illegible.remove(il)
                legible.remove(l)

    print(f"Duplicate found {duplicates_count}")
    illegibles_gt = [[x, 0] for x in illegible]
    legibles_gt = [[x, 1] for x in legible]
    all_gt = illegibles_gt + legibles_gt

    val_size = int(len(all_gt)*0.1)
    val_set = random.sample(all_gt, val_size)

    # create validation and training splits
    val_dir = os.path.join(dst, "val")
    val_images_dir = os.path.join(val_dir, "images")
    Path(val_images_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(val_dir, "val_gt.txt"), 'w') as vf:
        for entry in val_set:
            filename = os.path.basename(entry[0])
            vf.write(f'{filename},{entry[1]}\n')
            d = os.path.join(val_images_dir, filename)
            shutil.copyfile(entry[0], d)

    train_dir = os.path.join(dst, "train")
    train_images_dir = os.path.join(train_dir, "images")
    Path(train_images_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(train_dir, "train_gt.txt"), 'w') as tf:
        for entry in all_gt:
            if entry not in val_set:
                filename = os.path.basename(entry[0])
                tf.write(f'{filename},{entry[1]}\n')
                d = os.path.join(train_images_dir, filename)
                shutil.copyfile(entry[0], d)


#======================================================
# generate dataset to fine-tune number recognition

def get_track_number(img):
    tmp = img.split('_')
    return tmp[0]

def run_legibility_classifierrun_legibility_classifier(src, legible_json):
    soccer_net_gt_path = os.path.join(src, "train/train_gt.json")
    with open(soccer_net_gt_path, 'r') as f:
        soccer_net_gt = json.load(f)

    # run legibility for all track with jersey number, save full paths
    images_to_classify = []
    for key in soccer_net_gt.keys():
        if soccer_net_gt[key] != '-1':
            track_dir = os.path.join(src, 'train', 'images', key)
            images = os.listdir(track_dir)
            images_full_path = [os.path.join(track_dir, x) for x in images]
            images_to_classify += images_full_path

    if len(images_to_classify) == 0:
        return
    results = lc.run(images_to_classify, cfg.dataset['SoccerNet']['legibility_model'], threshold=0.6)
    legible = list(np.nonzero(results))[0]

    legible_images = [images_to_classify[i] for i in legible]

    # save results
    json_object = json.dumps(legible_images, indent=4)
    with open(legible_json, "w") as outfile:
        outfile.write(json_object)

def generate_jersey_number_dataset(src, dst, legible_path_json, split_by_track = 0.3):
    #get pose from legible images
    with open(legible_path_json, 'r') as f:
        legible_images = json.load(f)

    # for testing only:
    #    legible_images = legible_images[:20]
    ###

    input_json = os.path.join(dst, "pose_input.json")
    output_json = os.path.join(dst, "pose.json")
    #helpers.generate_json(legible_images, input_json)

    # print("Extracting pose")
    # command = f"conda run -n {cfg.pose_env} python3 pose.py {cfg.pose_home}/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py \
    #     {cfg.pose_home}/checkpoints/vitpose-h.pth --img-root / --json-file {input_json} \
    #     --out-json {output_json}"
    # success = os.system(command) == 0
    # if not success:
    #     print("Error extracting pose")

    print("Generate crops")
    crops_destination_dir = os.path.join(dst, "imgs")
    Path(crops_destination_dir).mkdir(parents=True, exist_ok=True)
    helpers.generate_crops(output_json, crops_destination_dir, legible_images, all_legible = legible_images)
    print("Done generating crops")

    # generate gt
    soccer_net_gt_path = os.path.join(src, "train/train_gt.json")
    with open(soccer_net_gt_path, 'r') as f:
        soccer_net_gt = json.load(f)
    all_images = os.listdir(crops_destination_dir)
    val_gt = []
    train_gt = []

    if split_by_track > 0:
        # convert to panda dataframe and add track column
        img_pd = pd.DataFrame(all_images, columns=['image'])
        img_pd['track'] = img_pd['image'].apply(helpers.get_track)
        track_numbers = img_pd.track.unique().size
        print(f'Found {track_numbers}, splitting {1 - split_by_track}/{split_by_track} into train/val')
        unique_tracks = set(img_pd.track.unique())
        val_tracks = random.sample(unique_tracks, int(track_numbers * split_by_track))
        for index, row in img_pd.iterrows():
            entry = [row['image'], soccer_net_gt[row['track']] ]
            if row['track'] in val_tracks:
                val_gt.append(entry)
            else:
                train_gt.append(entry)
    else:
        train_all = random.sample(all_images, int(len(all_images)*0.9))

        for img in all_images:
            track = get_track_number(img)
            label = soccer_net_gt[track]
            if img in train_all:
                train_gt.append([img, label])
            else:
                val_gt.append([img, label])

    train_gt_destination = os.path.join(dst, "train_gt.txt")
    val_gt_destination = os.path.join(dst, "val_gt.txt")

    with open(train_gt_destination, 'w') as tf:
        for entry in train_gt:
            tf.write(f"{entry[0]},{entry[1]}\n")

    with open(val_gt_destination, 'w') as vf:
        for entry in val_gt:
            vf.write(f"{entry[0]},{entry[1]}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', required=False, default='legibility', help="Options: 'legibility', 'numbers'")
    parser.add_argument('--src', required=True,  help='Location of data to weakly label')
    parser.add_argument('--dst', required=True,  help='Destination of generated weakly-labelled dataset')
    parser.add_argument('--legible_json', required=False, help='Used for jersey number dataset generation')
    parser.add_argument('--pose_json', required=False, help='Used for jersey number dataset generation')

    args = parser.parse_args()

    Path(args.dst).mkdir(parents=True, exist_ok=True)

    if args.type == 'legibility':
        generate_legibility_dataset(args.src, args.dst, use_backup=True)
    else:
        if not os.path.isfile(args.legible_json):
            run_legibility_classifier(args.src, args.legible_json)
        generate_jersey_number_dataset(args.src, args.dst, args.legible_json)
