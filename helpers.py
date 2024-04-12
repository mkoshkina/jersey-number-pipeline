import json
from copy import deepcopy
import cv2
import os
import json
import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd
import random
import shutil
from pathlib import Path
from scipy.special import softmax as softmax

json_img_template = { "id": 0,
            "file_name": "",
            "width": 0,
            "height": 0}

json_annotation_template = {"id": 0,
            "image_id": 0,
            "category_id": 1,
            "bbox": [
                0,
                0,
                0,
                0
            ]}

# Constants for pose-based torso cropping
PADDING = 5
CONFIDENCE_THRESHOLD = 0.4
TS = 2.367
HEIGHT_MIN = 35
WIDTH_MIN = 30

bias_for_digits = [0.06, 0.094, 0.094, 0.094, 0.094, 0.094, 0.094, 0.094, 0.094, 0.094, 0.094]

# Generate image JSON COCO format for ViTPose to consume
def generate_json(file_names, json_file_path):
    img_id = 0
    ann_id = 0
    json_dict = {"images": [], "annotations": []}
    for f in file_names:
        img = cv2.imread(f)
        height, width, _ = img.shape
        img_entry = deepcopy(json_img_template)
        ann_entry = deepcopy(json_annotation_template)

        img_entry["file_name"] = f
        img_entry["width"] = width
        img_entry["height"] = height
        img_entry["id"] = img_id

        ann_entry["id"] = ann_id
        ann_entry["image_id"] = img_id
        ann_entry["bbox"] = [0, 0, width, height]

        json_dict["images"].append(img_entry)
        json_dict["annotations"].append(ann_entry)

        img_id += 1

    with open(json_file_path, 'w') as fp:
        json.dump(json_dict, fp)

# get confidence-filtered points from pose results
def get_points(pose):
    points = pose["keypoints"]
    if len(points) < 12:
        #print("not enough points")
        return []
    relevant = [points[6], points[5], points[11], points[12]]
    result = []
    for r in relevant:
        if r[2] < CONFIDENCE_THRESHOLD:
            #print(f"confidence {r[2]}")
            return []
        result.append(r[:2])
    return result

# crop torso based on joints and save cropped images
def generate_crops_for_all(json_file, crops_destination_dir):
    with open(json_file, 'r') as f:
        all_poses = json.load(f)
        all_poses = all_poses["pose_results"]
    i = 0
    skipped = []
    saved = []
    for entry in tqdm(all_poses):
        filtered_points = get_points(entry)
        img_name = entry["img_name"]
        if len(filtered_points) == 0:
            #TODO: better approach then skipping
            print(f"skipping {img_name}")
            skipped.append(img_name)
            continue
        img = cv2.imread(img_name)
        if img is None:
            print(f"can't find {img_name}")
            continue
        height, width, _ = img.shape
        x_min = min([p[0] for p  in filtered_points]) - PADDING
        x_max = max([p[0] for p  in filtered_points]) + PADDING
        y_min = min([p[1] for p  in filtered_points]) - PADDING
        y_max = max([p[1] for p  in filtered_points])
        x1 = int(0 if x_min < 0 else x_min)
        y1 = int(0 if y_min < 0 else y_min)
        x2 = int(width - 1 if x_max > width else x_max)
        y2 = int(height -1 if y_max > height else y_max)
        #print(x1,x2,y1,y2)
        crop = img[y1:y2, x1:x2, :]
        h, w, _ = crop.shape
        if h == 0 or w == 0:
            print(f"skipping {img_name}")
            skipped.append(img_name)
            continue
        saved.append(img_name)
        temp = img_name.split('/')
        name = temp[-1]
        cv2.imwrite(os.path.join(crops_destination_dir, name), crop)
    return skipped, saved


def get_mean_conf(points):
    total = 0
    for p in points:
        total += p[2]
    return total/len(points)


def generate_crops_from_detections(det_path, crops_destination_dir, legible_results, images_dir):
    all_legible = []
    for key in legible_results.keys():
        for entry in legible_results[key]:
            all_legible.append(entry)
    with open(det_path, 'r') as f:
        bboxes = json.load(f)

    for img_name in all_legible:
        base_name = os.path.basename(img_name)
        if not base_name in bboxes.keys():
            continue
        det = bboxes[base_name]
        track = img_name.split('_')[0]
        img_path = os.path.join(images_dir, track, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"can't find {img_name}")
            continue
        height, width, _ = img.shape
        x_min = det[0] - PADDING
        x_max = det[2] + PADDING
        y_min = det[1] - PADDING
        y_max = det[3] + PADDING
        x1 = int(0 if x_min < 0 else x_min)
        y1 = int(0 if y_min < 0 else y_min)
        x2 = int(width - 1 if x_max > width else x_max)
        y2 = int(height -1 if y_max > height else y_max)
        #print(x1,x2,y1,y2)
        crop = img[y1:y2, x1:x2, :]
        h, w, _ = crop.shape

        cv2.imwrite(os.path.join(crops_destination_dir, base_name), crop)

# crop torso based on joints and save cropped images
def generate_crops(json_file, crops_destination_dir, legible_results, all_legible = None):
    if all_legible is None:
        all_legible = []
        for key in legible_results.keys():
            for entry in legible_results[key]:
                all_legible.append(os.path.basename(entry))
    with open(json_file, 'r') as f:
        all_poses = json.load(f)
        all_poses = all_poses["pose_results"]
    i = 0
    skipped = {}
    saved = []
    misses = 0
    for entry in tqdm(all_poses):
        filtered_points = get_points(entry)
        img_name = entry["img_name"]

        if not os.path.basename(img_name) in all_legible:
            continue
        if len(filtered_points) == 0:
            #TODO: better approach then skipping
            print(f"skipping {img_name}, unreliable points")
            tr = os.path.basename(img_name).split('_')[0]
            if tr not in skipped.keys():
                skipped[tr] = 1
            else:
                skipped[tr] += 1
            misses += 1
            continue
        img = cv2.imread(img_name)
        if img is None:
            print(f"can't find {img_name}")
            continue
        height, width, _ = img.shape
        x_min = min([p[0] for p  in filtered_points]) - PADDING
        x_max = max([p[0] for p  in filtered_points]) + PADDING
        y_min = min([p[1] for p  in filtered_points]) - PADDING
        y_max = max([p[1] for p  in filtered_points])
        x1 = int(0 if x_min < 0 else x_min)
        y1 = int(0 if y_min < 0 else y_min)
        x2 = int(width - 1 if x_max > width else x_max)
        y2 = int(height -1 if y_max > height else y_max)
        #print(x1,x2,y1,y2)
        crop = img[y1:y2, x1:x2, :]
        h, w, _ = crop.shape
        if h == 0 or w == 0:
            print(f"skipping {img_name}, shape is wrong")
            tr = os.path.basename(img_name).split('_')[0]
            if tr not in skipped.keys():
                skipped[tr] = 1
            else:
                skipped[tr] += 1
            misses += 1
            continue
        saved.append(img_name)
        name = os.path.basename(img_name)
        cv2.imwrite(os.path.join(crops_destination_dir, name), crop)
    print(f"skipped {misses} out of {len(all_poses)}")
    return skipped, saved

def is_valid_number(string):
    if string == '-' or len(string) > 2:
        return False
    try:
        num = int(string)
    except:
        return False
    if num > 0 and num < 100:
        return True
    else:
        False

# add bias - give twice the weight to double digit predictions
def get_bias(value):
    if int(value) > 9:
        return 0.61
    else:
        return 0.39

SUM_THRESHOLD = 1
FILTER_THRESHOLD = 0.2
def find_best_prediction(results, useBias=False):
    if FILTER_THRESHOLD > 0:
        for entry in results:
            if entry[1] < FILTER_THRESHOLD:
                entry[1] = 0
    unique_predictions = np.unique(results[:, 0])
    #print(unique_predictions)
    weights = []
    for i in range(len(unique_predictions)):
        value = unique_predictions[i]
        rows_with_value = results[np.where(results[:,0]==value)]
        b = get_bias(value) if useBias else 1
        adjusted_prob = rows_with_value[:, 1] * b
        sum_weights = np.sum(adjusted_prob)
        weights.append(sum_weights)

    best_weight = np.max(weights)
    index_of_best = np.argmax(weights)
    best_prediction = unique_predictions[index_of_best] if best_weight > SUM_THRESHOLD else -1
    return best_prediction, unique_predictions, weights


token_list = 'E0123456789'
# Test calibration
true_conf = [0.12, 0.13609467, 0.25098814, 0.38429752, 0.43631613, 0.48029819, 0.56463068, 0.94912186]
pred_conf = [0.27909853, 0.36124473, 0.45841138, 0.54969843, 0.65061644, 0.74988696, 0.85455219, 0.99713011]


def linear_interpolation(x1, y1, x2, y2, x):
    """
    Linear interpolation function to estimate y for a given x
    between two data points (x1, y1) and (x2, y2).
    """
    # Calculate the slope (m)
    m = (y2 - y1) / (x2 - x1)

    # Calculate the interpolated value (y)
    y = y1 + m * (x - x1)

    return y

def get_interval_index(prob):
    if prob < pred_conf[0]:
        return -1
    for i in range(8):
        if i == len(pred_conf)-1:
            return i
        if prob >= pred_conf[i] and prob <= pred_conf[i+1]:
            return i

def get_calibrated_value(prob):
    # find interval, linear function to calculate true prob
    i = get_interval_index(prob)
    if i == -1:
        return prob
    elif i == len(pred_conf)-1:
        return prob
    else:
        return linear_interpolation(pred_conf[i], true_conf[i], pred_conf[i+1], true_conf[i+1], prob)



def find_best_prediction_raw(results):
    mean_result = np.mean(results, axis=0)
    batch_tokens = ''
    batch_probs = []
    for dist in mean_result:
        id = np.argmax(dist)  # greedy selection
        prob = dist[id]
        batch_tokens += token_list[id]
        batch_probs.append(prob)
    print(len(batch_tokens), batch_tokens)
    for i in range(len(batch_tokens)):
        if batch_tokens[i] == 'E':
            batch_tokens = batch_tokens[:i]
            batch_probs = batch_probs[:i]
            break
    print(batch_tokens, batch_probs)
    return batch_tokens, batch_probs

def apply_bias(raw_result):
    marginal_likelihood = raw_result[1][0] * 0.39 + sum([x * 0.061 for x in raw_result[1][1:]])
    raw_result[1][0] = (raw_result[1][0] * 0.39) / marginal_likelihood
    raw_result[1][1:] = [(x * 0.061) / marginal_likelihood for x in raw_result[1][1:]]
    return raw_result

def calibrate_and_apply_bias_raw(raw_result):
    calib_function = np.vectorize(get_calibrated_value)
    raw_result = calib_function(raw_result)
    raw_result = apply_bias(raw_result)
    return raw_result

def find_best_prediction_with_vector(results):
    weights = np.sum(results, axis=0)

    best_weight = np.max(weights)
    index_of_best = np.argmax(weights)
    #best_prediction = unique_predictions[index_of_best] if best_weight > SUM_THRESHOLD else -1
    return index_of_best+1, [], weights


def apply_ts(logits):
    raw = np.array(logits) / TS
    conf0 = softmax(raw[0])
    conf1 = softmax(raw[1])
    return [conf0, conf1]


def initialize_priors(useBias, num_digits=11):
    """ Initialize uniform priors for each digit in both positions. """
    if not useBias:
        return np.full(num_digits, 1.0 / num_digits), np.full(num_digits, 1.0 / num_digits)
    else:
        return np.full(num_digits, 1.0 / num_digits), np.array(bias_for_digits)

def update_posteriors(priors, likelihoods):
    """ Update posterior probabilities based on the likelihoods (model outputs). """
    tens_priors, units_priors = priors
    tens_likelihood, units_likelihood = likelihoods

    # Update tens position
    tens_posterior = tens_priors * tens_likelihood
    tens_posterior /= np.sum(tens_posterior)

    # Update units position
    units_posterior = units_priors * units_likelihood
    units_posterior /= np.sum(units_posterior)

    return tens_posterior, units_posterior

def split_predictions_by_digit(image_predictions, priors=None):
    tens_likelihoods = []
    units_likelihoods = []
    for entry in image_predictions:
        e0 = entry[0]
        e1 = entry[1]
        if not (priors is None):
            e0, e1 = update_posteriors(priors, (e0, e1))
        tens_likelihoods.append(e0)
        units_likelihoods.append(e1)

    return np.array(tens_likelihoods), np.array(units_likelihoods)

def predict_jersey_number(image_predictions, useBias=False):
    """
    Predict the jersey number based on a sequence of image predictions.
    image_predictions: List of predictions for each image,
                       where each prediction is a tuple of two arrays
                       (tens and units likelihoods, each of size 10).
    """
    tens_priors, units_priors = initialize_priors(useBias)

    # use sum of log-likelihoods
    tens_likelihoods, units_likelihoods = split_predictions_by_digit(image_predictions, priors=(tens_priors, units_priors))

    # if useBias:
    #     units_likelihoods = units_likelihoods * bias_for_digits

    log_likelihoods_tens = np.log(tens_likelihoods)
    sum_logl_tens = np.sum(log_likelihoods_tens, axis=0)
    log_likelihoods_units = np.log(units_likelihoods)
    sum_logl_units = np.sum(log_likelihoods_units, axis=0)

    tens_digit = np.argmax(sum_logl_tens)
    units_digit = np.argmax(sum_logl_units)

    prob_tens = sum_logl_tens[tens_digit]
    prob_units = sum_logl_units[units_digit]

    batch_tokens = token_list[tens_digit] + token_list[units_digit]
    batch_probs = [prob_tens, prob_units]
    for i in range(2):
        if batch_tokens[i] == 'E':
            batch_tokens = batch_tokens[:i]
            batch_probs = batch_probs[:i]
            break

    return batch_tokens, batch_probs


def process_jersey_id_predictions_bayesian(file_path, useTS = False, useBias = False, useTh = False):
    all_results = {}
    final_results = {}
    with open(file_path, 'r') as f:
        results_dict = json.load(f)
    for name in results_dict.keys():
        tmp = name.split('_')
        tracklet = tmp[0]

        if tracklet not in all_results:
            all_results[tracklet] = []
            final_results[tracklet] = -1  # default
        if not useTS:
            raw_result = results_dict[name]['raw']
            #raw_result = calibrate_and_apply_bias_raw(raw_result)
            raw_result = np.array([np.array(xi) for xi in raw_result])
        else:
            raw_result = results_dict[name]['logits']
            raw_result = apply_ts(raw_result)
            #raw_result = apply_bias(raw_result)

        all_results[tracklet].append(raw_result)

    final_full_results = {}
    for tracklet in all_results.keys():
        if len(all_results[tracklet]) == 0:
            continue
        results = np.array(all_results[tracklet])

        best_prediction, probs = predict_jersey_number(results, useBias=useBias)

        # best_prediction, all_unique, weights = find_best_prediction_with_vector(results)
        prob = probs[0] if len(probs) == 1 else probs[0] + probs[1]
        if useTh and prob < -850:
                final_results[tracklet] = '-1'
                final_full_results[tracklet] = {'label': '-1', 'unique': [],
                                                'weights': probs}
        else:
            final_results[tracklet] = str(int(best_prediction))
            final_full_results[tracklet] = {'label': str(int(best_prediction)), 'unique': [],
                                        'weights': probs}

    return final_results, final_full_results

def process_jersey_id_predictions_raw(file_path, useTS = False ):
    all_results = {}
    final_results = {}
    with open(file_path, 'r') as f:
        results_dict = json.load(f)
    for name in results_dict.keys():
        tmp = name.split('_')
        tracklet = tmp[0]

        if tracklet not in all_results:
            all_results[tracklet] = []
            final_results[tracklet] = -1  # default
        if not useTS:
            raw_result = results_dict[name]['raw']
            #raw_result = calibrate_and_apply_bias_raw(raw_result)
            raw_result = np.array([np.array(xi) for xi in raw_result])
        else:
            raw_result = results_dict[name]['logits']
            raw_result = apply_ts(raw_result)
            raw_result = apply_bias(raw_result)

        all_results[tracklet].append(raw_result)

    final_full_results = {}
    for tracklet in all_results.keys():
        if len(all_results[tracklet]) == 0:
            continue
        results = np.array(all_results[tracklet])

        best_prediction, probs = find_best_prediction_raw(results)

        # best_prediction, all_unique, weights = find_best_prediction_with_vector(results)
        final_results[tracklet] = str(int(best_prediction))
        final_full_results[tracklet] = {'label': str(int(best_prediction)), 'unique': [],
                                        'weights': probs}

    return final_results, final_full_results

def identify_soccer_balls(image_dir, soccer_ball_list):
    # check 10 random images for each track, mark as soccer ball if the size matches typical soccer ball size
    ball_list = []
    tracklets = os.listdir(image_dir)
    for track in tqdm(tracklets):
        track_path = os.path.join(image_dir, track)
        image_names = os.listdir(track_path)
        sample = len(image_names) if len(image_names) < 10 else 10
        imgs = np.random.choice(image_names, size=sample, replace=False)
        width_list = []
        height_list = []
        for img_name in imgs:
            img_path = os.path.join(track_path, img_name)
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            width_list.append(w)
            height_list.append(h)
        mean_w, mean_h = np.mean(width_list), np.mean(height_list)
        if mean_h <= HEIGHT_MIN and mean_w <= WIDTH_MIN:
            # this must be a soccer ball
            ball_list.append(track)
    print(f"Found {len(ball_list)} balls, Ball list: {ball_list}")
    with open(soccer_ball_list, 'w') as fp:
        json.dump({'ball_tracks': ball_list}, fp)
    return True

def process_jersey_id_predictions(file_path, useBias=False):
    all_results = {}
    final_results = {}
    with open(file_path, 'r') as f:
        results_dict = json.load(f)
    for name in results_dict.keys():
        tmp = name.split('_')
        tracklet = tmp[0]

        if tracklet not in all_results:
            all_results[tracklet] = []
            final_results[tracklet] = -1 #default
        value = results_dict[name]['label']
        if not is_valid_number(value):
            continue
        confidence = results_dict[name]['confidence']
        # ingore last probability as it corresponds to 'end' token
        total_prob = 1
        for x in confidence[:-1]:
            total_prob = total_prob * float(x)

        all_results[tracklet].append([int(value), total_prob])

    final_full_results = {}
    for tracklet in all_results.keys():
        if len(all_results[tracklet]) == 0:
            continue
        results = np.array(all_results[tracklet])

        best_prediction, all_unique, weights = find_best_prediction(results, useBias=useBias)

        #best_prediction, all_unique, weights = find_best_prediction_with_vector(results)
        final_results[tracklet] = str(int(best_prediction))
        final_full_results[tracklet] = {'label':  str(int(best_prediction)), 'unique': all_unique, 'weights':weights}

    return final_results, final_full_results

THRESHOLD_FOR_TACK_LEGIBILITY = 0
def is_track_legible(track, illegible_list, legible_tracklets):
    if track in illegible_list:
        return False
    try:
        if len(legible_tracklets[track]) <= THRESHOLD_FOR_TACK_LEGIBILITY:
            return False
    except KeyError:
        return False
    return True

def evaluate_legibility(gt_path, illegible_path, legible_tracklets, soccer_ball_list = None):
    with open(gt_path, 'r') as gf:
        gt_dict = json.load(gf)
    with open(illegible_path, 'r') as gf:
        illegible_list = json.load(gf)
        illegible_list = illegible_list['illegible']

    balls_list = []
    if not soccer_ball_list is None:
        with open(soccer_ball_list, 'r') as sf:
            balls_json = json.load(sf)
        balls_list = balls_json['ball_tracks']

    correct = 0
    total = 0
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    num_per_tracklet_FP = []
    num_per_tracklet_TP = []
    for track in gt_dict.keys():
        # don't consider soccer balls
        if track in balls_list:
            continue

        true_value = str(gt_dict[track])
        predicted_legible = is_track_legible(track, illegible_list, legible_tracklets)
        if true_value == '-1' and not predicted_legible:
            #print(f"1){track}")
            correct += 1
            TN += 1
        elif true_value != '-1' and predicted_legible:
            #print(f"2){track}")
            correct += 1
            TP += 1
            # if legible_tracklets is not None:
            #     num_per_tracklet_TP.append(len(legible_tracklets[track]))
        elif true_value == '-1' and predicted_legible:
            FP += 1
            print(f"FP:{track}")
            # if legible_tracklets is not None:
            #     num_per_tracklet_FP.append(len(legible_tracklets[track]))
        elif true_value != '-1' and not predicted_legible:
            FN += 1
            print(f"FN:{track}")
        total += 1

    print(f'Correct {correct} out of {total}. Accuracy {100*correct/total}%.')
    print(f'TP={TP}, TN={TN}, FP={FP}, FN={FN}')
    Pr = TP / (TP + FP)
    Recall = TP / (TP + FN)
    print(f"Precision={Pr}, Recall={Recall}")
    print(f"F1={2 * Pr * Recall / (Pr + Recall)}")


SKIP_ILLEGIBLE = False
def evaluate_results(consolidated_dict, gt_dict, full_results = None):
    correct = 0
    total = 0
    mistakes = []
    count_of_correct_in_full_results = 0
    for id in gt_dict.keys():
        try:
            predicted = consolidated_dict[id]
        except KeyError:
            predicted = -1
            consolidated_dict[id] = -1
        if SKIP_ILLEGIBLE and (gt_dict[id] == -1 or predicted == -1):
            continue
        if str(gt_dict[id]) == str(predicted):
            correct += 1
        else:
            #print(predicted, gt_dict[id])
            mistakes.append(id)
        total += 1
    print(f"Total number of trackslets: {total}, correct: {correct}, accuracy: {100.0 * correct/total}%")
    #print(f"Tracklets with mistakes: {mistakes}")
    illegible_mistake_count = 0
    illegible_gt_count = 0
    for m in mistakes:
        #print(f"predicted:{consolidated_dict[m]},    real:{gt_dict[m]}")
        #count how many we considered illegible
        if str(consolidated_dict[m]) == str(-1):
            illegible_mistake_count += 1
        elif str(gt_dict[m]) == str(-1):
            illegible_gt_count += 1
        elif not (full_results is None):
            if m in full_results.keys():
                #print(f"track {m} , true label {gt_dict[m]}; predictions {full_results[m]}")
                if gt_dict[m] in full_results[m]['unique']:
                    count_of_correct_in_full_results += 1
        #print(f"track {m} , true label {gt_dict[m]}; prediction {consolidated_dict[m]}")
    #print(f'mismarked {illegible_mistake_count} out of {len(mistakes)} as illegible')
    #print(f'mismarked {illegible_gt_count} out of {len(mistakes)} as legible')
    #print(f"predicted correctly but not picked: {count_of_correct_in_full_results}")

def convert_polygon_to_bbox(polygon):
    # Initialize min and max values with the first vertex of the polygon.
    min_x, min_y = polygon[0], polygon[1]
    max_x, max_y = polygon[0], polygon[1]

    # Iterate through the vertices to find min and max coordinates.
    for i in range(0, len(polygon), 2):
        x, y = polygon[i], polygon[i + 1]
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)

    # Create the AABB as a tuple of (min_x, min_y, max_x, max_y).
    return [min_x, min_y, max_x, max_y]


def get_track(path):
    filename = os.path.basename(path)
    tmp = filename.split('_')
    return tmp[0]

def generate_different_split(current_directory, target_directory, split_val = 0.3):
    names = ['image', 'label']
    old_train_gt_path = os.path.join(current_directory, 'train', 'train_gt.txt')
    old_val_gt_path = os.path.join(current_directory, 'val', 'val_gt.txt')
    old_train_path = os.path.join(current_directory, 'train', 'images')
    old_val_path = os.path.join(current_directory, 'val', 'images')

    old_train_gt = pd.read_csv(old_train_gt_path, names=names, header=None)
    old_val_gt = pd.read_csv(old_val_gt_path, names=names, header=None)
    old_gt = pd.concat([old_train_gt, old_val_gt])

    def get_path(image_name):
        if image_name in set(old_train_gt['image']):
            dir = old_train_path
        else:
            dir = old_val_path
        return os.path.join(dir, image_name)

    old_gt['track'] = old_gt['image'].apply(get_track)
    track_numbers = old_gt.track.unique().size
    print(f'Found {track_numbers}, splitting {1-split_val}/{split_val} into train/val')
    unique_tracks = set(old_gt.track.unique())
    val_tracks = random.sample(unique_tracks, int(track_numbers*split_val))
    all_val_images = old_gt.loc[old_gt['track'].isin(val_tracks)]
    all_train_images = old_gt.loc[~old_gt['track'].isin(val_tracks)]

    # make directories if they don't exist
    new_val_dir = os.path.join(target_directory, 'val')
    new_train_dir = os.path.join(target_directory, 'train')

    new_val_img_dir = os.path.join(target_directory, 'val', 'images')
    new_train_img_dir = os.path.join(target_directory, 'train', 'images')

    Path(new_val_img_dir).mkdir(parents=True, exist_ok=True)
    Path(new_train_img_dir).mkdir(parents=True, exist_ok=True)

    #copy files and save new gt
    with open(os.path.join(new_val_dir, 'val_gt.txt'), 'w') as vf:
        for index, row in all_val_images.iterrows():
            # copy file
            filename = os.path.basename(row['image'])
            dst = os.path.join(new_val_img_dir, filename)
            shutil.copy(get_path(row['image']), dst)
            vf.write(f"{row['image']},{row['label']}\n")

    with open(os.path.join(new_train_dir, 'train_gt.txt'), 'w') as tf:
        for index, row in all_train_images.iterrows():
            # copy file
            filename = os.path.basename(row['image'])
            dst = os.path.join(new_train_img_dir, filename)
            shutil.copy(get_path(row['image']), dst)
            tf.write(f"{row['image']},{row['label']}\n")

pose_home = 'pose/ViTPose'
pose_env = 'vitpose'
def generate_crops_for_split(source, target, split):
    names = ['image', 'label']
    old_gt_path = os.path.join(source, split, split + '_gt.txt')
    old_gt = pd.read_csv(old_gt_path, names=names, header=None)
    old_path = os.path.join(source, split,  'images')

    legibles = old_gt[old_gt['label']==1]
    number_legibles = legibles.shape[0]
    sample_illegible = random.sample(set(old_gt[old_gt['label']==0].image),  number_legibles)
    all_images = []
    for img in legibles.image:
        all_images.append(os.path.join(old_path, img))
    for img in sample_illegible:
        all_images.append(os.path.join(old_path, img))

    print("Genetating json for pose detector")
    crops_dir = os.path.join(target, split, 'images')
    Path(crops_dir).mkdir(parents=True, exist_ok=True)
    input_json = os.path.join(target, f"pose_input_{split}.json")
    output_json = os.path.join(target, f"pose_{split}.json")
    generate_json(all_images, input_json)

    print("Extracting pose")
    command = f"conda run -n {pose_env} python3 pose.py {pose_home}/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py \
        {pose_home}/checkpoints/vitpose-h.pth --img-root / --json-file {input_json} \
        --out-json {output_json}"
    success = os.system(command) == 0
    if not success:
        print("Error extractivng pose")

    print("Generate crops")
    generate_crops_for_all(output_json, crops_dir)

    print("Update gt")
    # prune gt to remove images with no crops
    crops_list_train = os.listdir(crops_dir)
    with open(os.path.join(target, split, split+'_gt.txt'), 'w') as tf:
        for index, row in old_gt.iterrows():
            if row['image'] in crops_list_train:
                tf.write(f"{row['image']},{row['label']}\n")

def generate_crops_based(source, target, splits):
    for split in splits:
        print(f"Processing {split}")
        generate_crops_for_split(source, target, split)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', help="source directory of current legibility dataset")
    parser.add_argument('target', help="target directory for updated legibility dataset")
    parser.add_argument('--crops_based', action='store_true', default=False, help="extract crops for legibility dataset")
    parser.add_argument('--hockey', action='store_true', default=False,
                        help="update hockey dataset (otherwise soccer)")
    args = parser.parse_args()
    splits = ['train', 'val']
    if args.crops_based and args.hockey:
        splits.append('test')

    if not args.crops_based:
        generate_different_split(args.source, args.target)
    else:
        generate_crops_based(args.source, args.target, splits)


