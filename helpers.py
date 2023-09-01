import json
from copy import deepcopy
import cv2
import os
import json
import numpy as np

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
CALIBRATION_CONSTANT = 0.2

calibration_offsets = [0, 0.12764952, 0.181846,  0.22805455, 0.17882232, 0.19591278, 0.19673138, \
 0.27322199, 0.23161173, 0.1728571,  0.15665617, 0.21146313, 0.27525053,\
 0.26231429, 0.25607589, 0.25515485, 0.29957967, 0.26758864, 0.04985322]

def calibration_offset_lookup(confidence):
    if (confidence - calibration_offsets[int(20*confidence)-1]) < 0:
        return 0
    else:
        return calibration_offsets[int(20 * confidence) - 1]

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
    for entry in all_poses:
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

def analyze_pose_for_occlusion(pose_file):
    tracks = {'0':{}, '12':{}, '13':{}, '14':{}}
    with open(pose_file, 'r') as f:
        all_poses = json.load(f)
        all_poses = all_poses["pose_results"]

    for entry in all_poses:
        img_name = entry["img_name"]
        base_name = os.path.basename(img_name)
        track = base_name.split('_')[0]
        if not track in tracks.keys():
            continue
        mean_conf = get_mean_conf(entry["keypoints"])
        tracks[track][base_name] = mean_conf

    # sort
    for k in tracks.keys():
        current_track = tracks[k]
        myKeys = list(current_track.keys())
        myKeys.sort()
        sorted_dict = {i: current_track[i] for i in myKeys}
        tracks[k] = sorted_dict

    with open("/media/storage/jersey_ids/SoccerNetResults/occlusion_pose.json", 'w') as f:
        json.dump(tracks, f)

    print(tracks)

def filter_by_pose(pose_json, output_json, threshold=0.7):
    tracks = {}
    with open(pose_json, 'r') as f:
        all_poses = json.load(f)
        all_poses = all_poses["pose_results"]

    for entry in all_poses:
        img_name = entry["img_name"]
        base_name = os.path.basename(img_name)
        track = base_name.split('_')[0]
        mean_conf = get_mean_conf(entry["keypoints"])
        if not track in tracks.keys():
            tracks[track] = []
        if mean_conf <= threshold:
            continue
        tracks[track].append(base_name)

    with open(output_json, 'w') as of:
        json.dump(tracks, of)


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
def generate_crops(json_file, crops_destination_dir, legible_results):
    all_legible = []
    for key in legible_results.keys():
        for entry in legible_results[key]:
            all_legible.append(entry)
    with open(json_file, 'r') as f:
        all_poses = json.load(f)
        all_poses = all_poses["pose_results"]
    i = 0
    skipped = {}
    saved = []
    for entry in all_poses:
        filtered_points = get_points(entry)
        img_name = entry["img_name"]

        if not img_name in all_legible:
            continue
        if len(filtered_points) == 0:
            #TODO: better approach then skipping
            print(f"skipping {img_name}, unreliable points")
            tr = os.path.basename(img_name).split('_')[0]
            if tr not in skipped.keys():
                skipped[tr] = 1
            else:
                skipped[tr] += 1
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
            continue
        saved.append(img_name)
        temp = img_name.split('/')
        name = temp[-1]
        cv2.imwrite(os.path.join(crops_destination_dir, name), crop)
    print(skipped)
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

SUM_THRESHOLD = 0.5
FILTER_THRESHOLD = 0.2
def find_best_prediction(results):
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
        adjusted_prob = rows_with_value[:,1] * get_bias(value)
        sum_weights = np.sum(adjusted_prob)
        weights.append(sum_weights)

    best_weight = np.max(weights)
    index_of_best = np.argmax(weights)
    best_prediction = unique_predictions[index_of_best] if best_weight > SUM_THRESHOLD else -1
    return best_prediction, unique_predictions, weights


def find_best_prediction_with_vector(results):
    weights = np.sum(results, axis=0)

    best_weight = np.max(weights)
    index_of_best = np.argmax(weights)
    #best_prediction = unique_predictions[index_of_best] if best_weight > SUM_THRESHOLD else -1
    return index_of_best+1, [], weights

def convert_to_vector(value, probability):
    #probability *= get_bias(value)
    if probability > 1 or probability <= 0:
        print(probability)
    x = [(1 - probability) / 98 for i in range(99)]
    x[value-1] = probability
    x = np.log(x)
    # add bias
    x[:8] = x[:8] + np.log(0.39/9)
    x[8:] = x[8:] + np.log(0.61/90)
    return x


def process_jersey_id_predictions(file_path):
    all_results = {}
    final_results = {}
    with open(file_path, 'r') as f:
        results_dict = json.load(f)
    for name in results_dict.keys():
        tmp = name.split('_')
        tracklet = tmp[0]

        if tracklet not in all_results:
            all_results[tracklet] = []
            final_results[tracklet] = -1 # default
        value = results_dict[name]['label']
        if not is_valid_number(value):
            continue
        confidence = results_dict[name]['confidence']
        # ingore last probability as it corresponds to 'end' token
        total_prob = 1
        for x in confidence[:-1]:
            total_prob = total_prob * float(x)

        all_results[tracklet].append([int(value), total_prob-CALIBRATION_CONSTANT])
        #all_results[tracklet].append(convert_to_vector(int(value), total_prob - calibration_offset_lookup(total_prob)))

    final_full_results = {}
    for tracklet in all_results.keys():
        if len(all_results[tracklet]) == 0:
            continue
        results = np.array(all_results[tracklet])

        best_prediction, all_unique, weights = find_best_prediction(results)

        #best_prediction, all_unique, weights = find_best_prediction_with_vector(results)
        final_results[tracklet] = str(int(best_prediction))
        final_full_results[tracklet] = {'label':  str(int(best_prediction)), 'unique': all_unique, 'weights':weights}

    return final_results, final_full_results

THRESHOLD_FOR_TACK_LEGIBILITY = 0
def is_track_legible(track, illegible_list, legible_tracklets):
    if track in illegible_list:
        return False
    if len(legible_tracklets[track]) <= THRESHOLD_FOR_TACK_LEGIBILITY:
        return False
    return True

def evaluate_legibility(gt_path, illegible_path, legible_tracklets):
    with open(gt_path, 'r') as gf:
        gt_dict = json.load(gf)
    with open(illegible_path, 'r') as gf:
        illegible_list = json.load(gf)
        illegible_list = illegible_list['illegible']
    correct = 0
    total = 0
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    num_per_tracklet_FP = []
    num_per_tracklet_TP = []
    for track in gt_dict.keys():
        true_value = str(gt_dict[track])
        if true_value == '-1' and not is_track_legible(track, illegible_list, legible_tracklets):
            correct += 1
            TN += 1
        elif true_value != '-1' and is_track_legible(track, illegible_list, legible_tracklets):
            correct += 1
            TP += 1
            # if legible_tracklets is not None:
            #     num_per_tracklet_TP.append(len(legible_tracklets[track]))
        elif true_value == '-1' and is_track_legible(track, illegible_list, legible_tracklets):
            FP += 1
            print(track, true_value)
            # if legible_tracklets is not None:
            #     num_per_tracklet_FP.append(len(legible_tracklets[track]))
        elif true_value != '-1' and not is_track_legible(track, illegible_list, legible_tracklets):
            FN += 1
            print(track, true_value)
        total += 1

    print(f'Correct {correct} out of {total}. Accuracy {100*correct/total}%.')
    print(f'TP={TP}, TN={TN}, FP={FP}, FN={FN}')
    #print(f"FP average num images: {np.mean(num_per_tracklet_FP)} \n Distribution: {num_per_tracklet_FP}")
    #print(f"TP average num images: {np.mean(num_per_tracklet_TP)} \n Distribution: {num_per_tracklet_TP}")


def find_best_threshold(legible_results, gt_path):
    with open(gt_path, 'r') as gf:
        gt_dict = json.load(gf)
    thresholds = [0.1 * x for x in range(10)]
    all_correct = []
    all_fn = []
    all_fp = []
    for th in thresholds:
        correct = 0
        FP = 0
        FN = 0
        for tracklet in legible_results.keys():
            track_results = (np.array(legible_results[tracklet])>th)
            label = 0 if track_results.sum() == 0 else 1
            if label == 0 and gt_dict[tracklet] == -1 or label == 1 and gt_dict[tracklet] > 0:
                correct += 1
            elif label == 1 and gt_dict[tracklet] == -1:
                FP += 1
            elif label == 0 and gt_dict[tracklet] > 0:
                FN += 1

        all_correct.append(correct)
        all_fn.append(FN)
        all_fp.append(FP)
        print(f"Threshold {th} gives {correct} correct labels out of {len(legible_results.keys())}")

    best_acc_indx = np.argmax(all_correct)
    full_results = {'correct':all_correct, 'FN': all_fn, 'FP': all_fp}

    return thresholds[best_acc_indx], all_correct[best_acc_indx], full_results


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
                print(f"track {m} , true label {gt_dict[m]}; predictions {full_results[m]}")
                if gt_dict[m] in full_results[m]['unique']:
                    count_of_correct_in_full_results += 1
        print(f"track {m} , true label {gt_dict[m]}; prediction {consolidated_dict[m]}")
    print(f'mismarked {illegible_mistake_count} out of {len(mistakes)} as illegible')
    print(f'mismarked {illegible_gt_count} out of {len(mistakes)} as legible')
    print(f"predicted correctly but not picked: {count_of_correct_in_full_results}")

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


def evaluate_pose_estimation(input_json, crops_destination_dir):
    all_crops = os.listdir(crops_destination_dir)
    all_tracklets_in_crops = set([])
    for crop in all_crops:
        filename = os.path.basename(crop)
        tmp = filename.split('_')
        all_tracklets_in_crops.add(tmp[0])
    with open(input_json, 'r') as f:
        input = json.load(f)

    input_tracklets = input.keys()

    total_legible = len(input_tracklets)
    missing = total_legible - len(all_tracklets_in_crops)

    print(f"Missing {missing} tracklets out of {total_legible}")


