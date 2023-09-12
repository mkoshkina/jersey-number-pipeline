import argparse
import os
import legibility_classifier as lc
import numpy as np
import json
import helpers
from tqdm import tqdm
import configuration as config
from pathlib import Path

def get_soccer_net_raw_legibility_results(args):
    root_dir = config.dataset['SoccerNet']['root_dir']
    image_dir = config.dataset['SoccerNet'][args.part]['images']
    path_to_images = os.path.join(root_dir, image_dir)
    tracklets = os.listdir(path_to_images)
    results_dict = {x:[] for x in tracklets}

    for directory in tqdm(tracklets):
        track_dir = os.path.join(path_to_images, directory)
        images = os.listdir(track_dir)
        images_full_path = [os.path.join(track_dir, x) for x in images]
        track_results = lc.run(images_full_path, config.dataset['SoccerNet']['legibility_model'], threshold=-1)
        results_dict[directory] = track_results

    # save results
    full_legibile_path = os.path.join(config.dataset['SoccerNet']['working_dir'], config.dataset['SoccerNet']['raw_legible_result'])
    with open(full_legibile_path, "w") as outfile:
        json.dump(results_dict, outfile)

    return results_dict

def get_soccer_net_legibility_results(args, use_filtered = False, filter = 'sim'):
    root_dir = config.dataset['SoccerNet']['root_dir']
    image_dir = config.dataset['SoccerNet'][args.part]['images']
    path_to_images = os.path.join(root_dir, image_dir)
    tracklets = os.listdir(path_to_images)

    if use_filtered:
        if filter == 'sim':
            path_to_filter_results = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                                  config.dataset['SoccerNet']['sim_filtered'])
        else:
            path_to_filter_results = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                                  config.dataset['SoccerNet']['gauss_filtered'])
        with open(path_to_filter_results, 'r') as f:
            filtered = json.load(f)

    legible_tracklets = {}
    illegible_tracklets = []

    for directory in tqdm(tracklets):
        track_dir = os.path.join(path_to_images, directory)
        if use_filtered:
            images = filtered[directory]
        else:
            images = os.listdir(track_dir)
        images_full_path = [os.path.join(track_dir, x) for x in images]
        track_results = lc.run(images_full_path, config.dataset['SoccerNet']['legibility_model'], threshold=0.8)
        legible = list(np.nonzero(track_results))[0]
        if len(legible) == 0:
            illegible_tracklets.append(directory)
        else:
            legible_images = [images_full_path[i] for i in legible]
            legible_tracklets[directory] = legible_images

    # save results
    json_object = json.dumps(legible_tracklets, indent=4)
    full_legibile_path = os.path.join(config.dataset['SoccerNet']['working_dir'], config.dataset['SoccerNet']['legible_result'])
    with open(full_legibile_path, "w") as outfile:
        outfile.write(json_object)

    full_illegibile_path = os.path.join(config.dataset['SoccerNet']['working_dir'], config. dataset['SoccerNet']['illegible_result'])
    json_object = json.dumps({'illegible': illegible_tracklets}, indent=4)
    with open(full_illegibile_path, "w") as outfile:
        outfile.write(json_object)

    return legible_tracklets, illegible_tracklets

def generate_json_for_pose_estimator(args, legible = None):
    all_files = []
    if not legible is None:
        for key in legible.keys():
            for entry in legible[key]:
                all_files.append(os.path.join(os.getcwd(), entry))
    else:
        root_dir = os.path.join(os.getcwd(), config.dataset['SoccerNet']['root_dir'])
        image_dir = config.dataset['SoccerNet'][args.part]['images']
        path_to_images = os.path.join(root_dir, image_dir)
        tracks = os.listdir(path_to_images)
        for tr in tracks:
            track_dir = os.path.join(path_to_images, tr)
            imgs = os.listdir(track_dir)
            for img in imgs:
                all_files.append(os.path.join(track_dir, img))

    output_json = os.path.join(config.dataset['SoccerNet']['working_dir'], config.dataset['SoccerNet']['pose_input_json'])
    helpers.generate_json(all_files, output_json)


def consolidated_results(dict, illegible_path):
    with open(illegible_path, 'r') as f:
        illegile_dict = json.load(f)
    all_illegible = illegile_dict['illegible']
    for entry in all_illegible:
        dict[str(entry)] = -1
    return dict


def hockey_pipeline(args):
    # Code to be released later
    print("Code to be released")

def soccer_net_pipeline(args):
    legible_dict = None
    legible_results = None
    consolidated_dict = None
    Path(config.dataset['SoccerNet']['working_dir']).mkdir(parents=True, exist_ok=True)
    success = True

    image_dir = os.path.join(config.dataset['SoccerNet']['root_dir'], config.dataset['SoccerNet']['test']['images'])
    features_dir = config.dataset['SoccerNet']['test']['feature_output_folder']
    full_legibile_path = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                      config.dataset['SoccerNet']['legible_result'])
    illegible_path = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                  config.dataset['SoccerNet']['illegible_result'])
    gt_path = os.path.join(config.dataset['SoccerNet']['root_dir'], config.dataset['SoccerNet']['test']['gt'])

    input_json = os.path.join(config.dataset['SoccerNet']['working_dir'],
                              config.dataset['SoccerNet']['pose_input_json'])
    output_json = os.path.join(config.dataset['SoccerNet']['working_dir'],
                               config.dataset['SoccerNet']['pose_output_json'])

    # 1. generate and store features for each image in each tracklet
    if args.pipeline['feat']:
        print("Generate features")
        command = f"conda run -n {config.reid_env} python3 centroid_reid.py --tracklets_folder {image_dir} --output_folder {features_dir}"
        success = os.system(command) == 0
        print("Done generating features")

    #2. identify and remove outliers based on features
    if args.pipeline['filter'] and success:
        print("Identify and remove outliers")
        command = f"python3 gaussian_outliers.py --tracklets_folder {image_dir} --output_folder {features_dir}"
        success = os.system(command) == 0
        print("Done removing outliers")

    #3. pass all images through legibililty classifier and record results
    if args.pipeline['legible'] and success:
        print("Classifying Legibility:")
        try:
            legible_dict, illegible_tracklets = get_soccer_net_legibility_results(args, use_filtered=args.pipeline['filter'], filter='gauss')
        except:
            success = False
        print("Done classifying legibility")

    #3.5 evaluate tracklet legibility results
    if args.pipeline['legible_eval'] and success:
        print("Evaluate Legibility results:")
        try:
            if legible_dict is None:
                 with open(full_legibile_path, 'r') as openfile:
                    # Reading from json file
                    legible_dict = json.load(openfile)

            helpers.evaluate_legibility(gt_path, illegible_path, legible_dict)
        except Exception as e:
            print(e)
            success = False
        print("Done evaluating legibility")


    #4. generate json for pose-estimation
    if args.pipeline['pose'] and success:
        print("Generating json for pose")
        try:
            if legible_dict is None:
                with open(full_legibile_path, 'r') as openfile:
                    # Reading from json file
                    legible_dict = json.load(openfile)
            generate_json_for_pose_estimator(args, legible = legible_dict)
        except Exception as e:
            print(e)
            success = False
        print("Done generating json for pose")

        # 4.5 Alternatively generate json for pose for all images in test/train
        #generate_json_for_pose_estimator(args)


        #5. run pose estimation and store results
        if success:
            print("Detecting pose")
            command = f"conda run -n {config.pose_env} python3 pose.py {config.pose_home}/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py \
                {config.pose_home}/checkpoints/vitpose-h.pth --img-root / --json-file {input_json} \
                --out-json {output_json}"
            success = os.system(command) == 0
            print("Done detecting pose")


    #6. generate cropped images
    if args.pipeline['crops'] and success:
        print("Generate crops")
        try:
            crops_destination_dir = os.path.join(config.dataset['SoccerNet']['working_dir'], config.dataset['SoccerNet']['crops_folder'], 'imgs')
            Path(crops_destination_dir).mkdir(parents=True, exist_ok=True)
            if legible_results is None:
                with open(full_legibile_path, "r") as outfile:
                    legible_results = json.load(outfile)
            helpers.generate_crops(output_json, crops_destination_dir, legible_results)
        except Exception as e:
            print(e)
            success = False
        print("Done generating crops")


    #7. run STR system on all crops
    if args.pipeline['str'] and success:
        print("Predict numbers")
        image_dir = os.path.join(config.dataset['SoccerNet']['working_dir'], config.dataset['SoccerNet']['crops_folder'])
        str_result_file = os.path.join(config.dataset['SoccerNet']['working_dir'], config.dataset['SoccerNet']['jersey_id_result'])
        command = f"conda run -n {config.str_env} python3 str.py  {config.dataset['SoccerNet']['str_model']}\
            --data_root={image_dir} --batch_size=1 --inference --result_file {str_result_file}"
        success = os.system(command) == 0
        print("Done predict numbers")

    if args.pipeline['eval'] and success:
        #8. combine tracklet results
        analysis_results = None
        #read predicted results, stack unique predictions, sum confidence scores for each, choose argmax
        results_dict, analysis_results = helpers.process_jersey_id_predictions(str_result_file)

        # add illegible tracklet predictions
        consolidated_dict = consolidated_results(results_dict, illegible_path)

        #save results as json
        final_results_path = os.path.join(config.dataset['SoccerNet']['working_dir'], config.dataset['SoccerNet']['final_result'])
        with open(final_results_path, 'w') as f:
            json.dump(consolidated_dict, f)

        #9. evaluate accuracy
        if consolidated_dict is None:
            with open(final_results_path, 'r') as f:
                consolidated_dict = json.load(f)
        with open(gt_path, 'r') as gf:
            gt_dict = json.load(gf)
        print(len(consolidated_dict.keys()), len(gt_dict.keys()))
        helpers.evaluate_results(consolidated_dict, gt_dict, full_results = analysis_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help="Options: 'SoccerNet', 'Hockey'")
    parser.add_argument('part', help="Options: 'test', 'val', 'train'")
    args = parser.parse_args()
    actions = {'feat': False,
               "filter": False,
               "legible": False,
               "legible_eval": False,
               "pose": False,
               "crops": False,
               "str": True,
               "eval": True}
    args.pipeline = actions
    if args.dataset == 'SoccerNet':
        soccer_net_pipeline(args)
    elif args.dataset == 'Hockey':
        hockey_pipeline(args)
    else:
        print("Unknown dataset")


