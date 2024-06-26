import argparse
import os
import legibility_classifier as lc
import numpy as np
import json
import helpers
from tqdm import tqdm
import configuration as config
from pathlib import Path

def get_soccer_net_raw_legibility_results(args, use_filtered = True, filter = 'gauss', exclude_balls=True):
    root_dir = config.dataset['SoccerNet']['root_dir']
    image_dir = config.dataset['SoccerNet'][args.part]['images']
    path_to_images = os.path.join(root_dir, image_dir)
    tracklets = os.listdir(path_to_images)
    results_dict = {x:[] for x in tracklets}

    if use_filtered:
        if filter == 'sim':
            path_to_filter_results = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                                  config.dataset['SoccerNet'][args.part]['sim_filtered'])
        else:
            path_to_filter_results = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                                  config.dataset['SoccerNet'][args.part]['gauss_filtered'])
        with open(path_to_filter_results, 'r') as f:
            filtered = json.load(f)


    if exclude_balls:
        updated_tracklets = []
        soccer_ball_list = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                        config.dataset['SoccerNet'][args.part]['soccer_ball_list'])
        with open(soccer_ball_list, 'r') as f:
            ball_json = json.load(f)
        ball_list = ball_json['ball_tracks']
        for track in tracklets:
            if not track in ball_list:
                updated_tracklets.append(track)
        tracklets = updated_tracklets

    for directory in tqdm(tracklets):
        track_dir = os.path.join(path_to_images, directory)
        if use_filtered:
            images = filtered[directory]
        else:
            images = os.listdir(track_dir)
        #images = os.listdir(track_dir)
        images_full_path = [os.path.join(track_dir, x) for x in images]
        track_results = lc.run(images_full_path, config.dataset['SoccerNet']['legibility_model'], threshold=-1, arch=config.dataset['SoccerNet']['legibility_model_arch'])
        results_dict[directory] = track_results

    # save results
    full_legibile_path = os.path.join(config.dataset['SoccerNet']['working_dir'], config.dataset['SoccerNet'][args.part]['raw_legible_result'])
    with open(full_legibile_path, "w") as outfile:
        json.dump(results_dict, outfile)

    return results_dict

def get_soccer_net_legibility_results(args, use_filtered = False, filter = 'sim', exclude_balls=True):
    root_dir = config.dataset['SoccerNet']['root_dir']
    image_dir = config.dataset['SoccerNet'][args.part]['images']
    path_to_images = os.path.join(root_dir, image_dir)
    tracklets = os.listdir(path_to_images)

    if use_filtered:
        if filter == 'sim':
            path_to_filter_results = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                                  config.dataset['SoccerNet'][args.part]['sim_filtered'])
        else:
            path_to_filter_results = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                                  config.dataset['SoccerNet'][args.part]['gauss_filtered'])
        with open(path_to_filter_results, 'r') as f:
            filtered = json.load(f)

    legible_tracklets = {}
    illegible_tracklets = []

    if exclude_balls:
        updated_tracklets = []
        soccer_ball_list = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                        config.dataset['SoccerNet'][args.part]['soccer_ball_list'])
        with open(soccer_ball_list, 'r') as f:
            ball_json = json.load(f)
        ball_list = ball_json['ball_tracks']
        for track in tracklets:
            if not track in ball_list:
                updated_tracklets.append(track)
        tracklets = updated_tracklets


    for directory in tqdm(tracklets):
        track_dir = os.path.join(path_to_images, directory)
        if use_filtered:
            images = filtered[directory]
        else:
            images = os.listdir(track_dir)
        images_full_path = [os.path.join(track_dir, x) for x in images]
        track_results = lc.run(images_full_path, config.dataset['SoccerNet']['legibility_model'], arch=config.dataset['SoccerNet']['legibility_model_arch'], threshold=0.5)
        legible = list(np.nonzero(track_results))[0]
        if len(legible) == 0:
            illegible_tracklets.append(directory)
        else:
            legible_images = [images_full_path[i] for i in legible]
            legible_tracklets[directory] = legible_images

    # save results
    json_object = json.dumps(legible_tracklets, indent=4)
    full_legibile_path = os.path.join(config.dataset['SoccerNet']['working_dir'], config.dataset['SoccerNet'][args.part]['legible_result'])
    with open(full_legibile_path, "w") as outfile:
        outfile.write(json_object)

    full_illegibile_path = os.path.join(config.dataset['SoccerNet']['working_dir'], config. dataset['SoccerNet'][args.part]['illegible_result'])
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

    output_json = os.path.join(config.dataset['SoccerNet']['working_dir'], config.dataset['SoccerNet'][args.part]['pose_input_json'])
    helpers.generate_json(all_files, output_json)


def consolidated_results(image_dir, dict, illegible_path, soccer_ball_list=None):
    if not soccer_ball_list is None:
        with open(soccer_ball_list, 'r') as sf:
            balls_json = json.load(sf)
        balls_list = balls_json['ball_tracks']
        for entry in balls_list:
            dict[str(entry)] = 1

    with open(illegible_path, 'r') as f:
        illegile_dict = json.load(f)
    all_illegible = illegile_dict['illegible']
    for entry in all_illegible:
        if not str(entry) in dict.keys():
            dict[str(entry)] = -1

    all_tracks = os.listdir(image_dir)
    for t in all_tracks:
        if not t in dict.keys():
            dict[t] = -1
        else:
            dict[t] = int(dict[t])
    return dict

def train_parseq(args):
    if args.dataset == 'Hockey':
        print("Train PARSeq for Hockey")
        parseq_dir = config.str_home
        current_dir = os.getcwd()
        os.chdir(parseq_dir)
        data_root = os.path.join(current_dir, config.dataset['Hockey']['root_dir'], config.dataset['Hockey']['numbers_data'])
        command = f"conda run -n {config.str_env} python3 train.py +experiment=parseq dataset=real data.root_dir={data_root} trainer.max_epochs=25 " \
                  f"pretrained=parseq trainer.devices=1 trainer.val_check_interval=1 data.batch_size=128 data.max_label_length=2"
        success = os.system(command) == 0
        os.chdir(current_dir)
        print("Done training")
    else:
        print("Train PARSeq for Soccer")
        parseq_dir = config.str_home
        current_dir = os.getcwd()
        os.chdir(parseq_dir)
        data_root = os.path.join(current_dir, config.dataset['SoccerNet']['root_dir'], config.dataset['SoccerNet']['numbers_data'])
        command = f"conda run -n {config.str_env} python3 train.py +experiment=parseq dataset=real data.root_dir={data_root} trainer.max_epochs=25 " \
                  f"pretrained=parseq trainer.devices=1 trainer.val_check_interval=1 data.batch_size=128 data.max_label_length=2"
        success = os.system(command) == 0
        os.chdir(current_dir)
        print("Done training")


def hockey_pipeline(args):
    # actions = {"legible": True,
    #            "pose": False,
    #            "crops": False,
    #            "str": True}
    success = True
    # test legibility classification
    if args.pipeline['legible']:
        root_dir = os.path.join(config.dataset["Hockey"]["root_dir"], config.dataset["Hockey"]["legibility_data"])

        print("Test legibility classifier")
        command = f"python3 legibility_classifier.py --data {root_dir} --arch resnet34 --trained_model {config.dataset['Hockey']['legibility_model']}"
        success = os.system(command) == 0
        print("Done legibility classifier")

    if success and args.pipeline['str']:
        print("Predict numbers")
        current_dir = os.getcwd()
        data_root = os.path.join(current_dir, config.dataset['Hockey']['root_dir'], config.dataset['Hockey']['numbers_data'])
        command = f"conda run -n {config.str_env} python3 str.py  {config.dataset['Hockey']['str_model']}\
            --data_root={data_root}"
        success = os.system(command) == 0
        print("Done predict numbers")

def soccer_net_pipeline(args):
    legible_dict = None
    legible_results = None
    consolidated_dict = None
    Path(config.dataset['SoccerNet']['working_dir']).mkdir(parents=True, exist_ok=True)
    success = True

    image_dir = os.path.join(config.dataset['SoccerNet']['root_dir'], config.dataset['SoccerNet'][args.part]['images'])
    soccer_ball_list = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                      config.dataset['SoccerNet'][args.part]['soccer_ball_list'])
    features_dir = config.dataset['SoccerNet'][args.part]['feature_output_folder']
    full_legibile_path = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                      config.dataset['SoccerNet'][args.part]['legible_result'])
    illegible_path = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                  config.dataset['SoccerNet'][args.part]['illegible_result'])
    gt_path = os.path.join(config.dataset['SoccerNet']['root_dir'], config.dataset['SoccerNet'][args.part]['gt'])

    input_json = os.path.join(config.dataset['SoccerNet']['working_dir'],
                              config.dataset['SoccerNet'][args.part]['pose_input_json'])
    output_json = os.path.join(config.dataset['SoccerNet']['working_dir'],
                               config.dataset['SoccerNet'][args.part]['pose_output_json'])

    # 1. Filter out soccer ball based on images size
    if args.pipeline['soccer_ball_filter']:
        print("Determine soccer ball")
        success = helpers.identify_soccer_balls(image_dir, soccer_ball_list)
        print("Done determine soccer ball")

    # 1. generate and store features for each image in each tracklet
    if args.pipeline['feat']:
        print("Generate features")
        command = f"conda run -n {config.reid_env} python3 {config.reid_script} --tracklets_folder {image_dir} --output_folder {features_dir}"
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
            legible_dict, illegible_tracklets = get_soccer_net_legibility_results(args, use_filtered=True, filter='gauss', exclude_balls=True)
            #get_soccer_net_raw_legibility_results(args)
            #legible_dict, illegible_tracklets = get_soccer_net_combined_legibility_results(args)
        except Exception as error:
            print(f'Failed to run legibility classifier:{error}')
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

            helpers.evaluate_legibility(gt_path, illegible_path, legible_dict, soccer_ball_list=soccer_ball_list)
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
            crops_destination_dir = os.path.join(config.dataset['SoccerNet']['working_dir'], config.dataset['SoccerNet'][args.part]['crops_folder'], 'imgs')
            Path(crops_destination_dir).mkdir(parents=True, exist_ok=True)
            if legible_results is None:
                with open(full_legibile_path, "r") as outfile:
                    legible_results = json.load(outfile)
            helpers.generate_crops(output_json, crops_destination_dir, legible_results)
        except Exception as e:
            print(e)
            success = False
        print("Done generating crops")

    str_result_file = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                   config.dataset['SoccerNet'][args.part]['jersey_id_result'])
    #7. run STR system on all crops
    if args.pipeline['str'] and success:
        print("Predict numbers")
        image_dir = os.path.join(config.dataset['SoccerNet']['working_dir'], config.dataset['SoccerNet'][args.part]['crops_folder'])

        command = f"conda run -n {config.str_env} python3 str.py  {config.dataset['SoccerNet']['str_model']}\
            --data_root={image_dir} --batch_size=1 --inference --result_file {str_result_file}"
        success = os.system(command) == 0
        print("Done predict numbers")

    #str_result_file = os.path.join(config.dataset['SoccerNet']['working_dir'], "val_jersey_id_predictions.json")
    if args.pipeline['combine'] and success:
        #8. combine tracklet results
        analysis_results = None
        #read predicted results, stack unique predictions, sum confidence scores for each, choose argmax
        results_dict, analysis_results = helpers.process_jersey_id_predictions(str_result_file, useBias=True)
        #results_dict, analysis_results = helpers.process_jersey_id_predictions_raw(str_result_file, useTS=True)
        #results_dict, analysis_results = helpers.process_jersey_id_predictions_bayesian(str_result_file, useTS=True, useBias=True, useTh=True)

        # add illegible tracklet predictions
        consolidated_dict = consolidated_results(image_dir, results_dict, illegible_path, soccer_ball_list=soccer_ball_list)

        #save results as json
        final_results_path = os.path.join(config.dataset['SoccerNet']['working_dir'], config.dataset['SoccerNet'][args.part]['final_result'])
        with open(final_results_path, 'w') as f:
            json.dump(consolidated_dict, f)

    if args.pipeline['eval'] and success:
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
    parser.add_argument('part', help="Options: 'test', 'val', 'train', 'challenge")
    parser.add_argument('--train_str', action='store_true', default=False, help="Run training of jersey number recognition")
    args = parser.parse_args()

    if not args.train_str:
        if args.dataset == 'SoccerNet':
            actions = {"soccer_ball_filter": True,
                       "feat": True,
                       "filter": True,
                       "legible": True,
                       "legible_eval": False,
                       "pose": True,
                       "crops": True,
                       "str": True,
                       "combine": True,
                       "eval": True}
            args.pipeline = actions
            soccer_net_pipeline(args)
        elif args.dataset == 'Hockey':
            actions = {"legible": True,
                       "str": True}
            args.pipeline = actions
            hockey_pipeline(args)
        else:
            print("Unknown dataset")
    else:
        train_parseq(args)


