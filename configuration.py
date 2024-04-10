pose_home = 'pose/ViTPose'
pose_env = 'vitpose'

str_home = 'str/parseq/'
str_env = 'parseq2'
str_platform = 'cu113'

# centroids
#reid_env = 'centroids'
#reid_script = 'centroid_reid.py'

#fastereid (soccernet fine-tuned)
reid_env = 'SUSHI'
reid_script = 'fast_reid.py'

reid_home = 'reid/'


dataset = {'SoccerNet':
                {'root_dir': './data/SoccerNet',
                 'working_dir': './out/SoccerNetResults',
                 'test': {
                        'images': 'test/images',
                        #'gt': 'test/test_gt_updated.json',
                        'gt': 'test/test_gt.json',
                        #'feature_output_folder': 'out/SoccerNetResults/fastreid_feat_test',
                        'feature_output_folder': 'out/SoccerNetResults/test',
                        'illegible_result': 'illegible.json',
                        #'illegible_result': 'illegible_no_f.json',
                        'soccer_ball_list': 'soccer_ball.json',
                        'sim_filtered': 'test/main_subject_0.4.json',
                        'gauss_filtered': 'test/main_subject_gauss_th=3.5_r=3.json',
                        #'gauss_filtered': 'fastreid_feat_test/main_subject_gauss_th=3.5_r=3.json',
                        'legible_result': 'legible.json',
                        #'legible_result': 'legible_no_f.json',
                        'raw_legible_result': 'raw_legible_resnet34.json',
                        'pose_input_json': 'pose_input.json',
                        'pose_output_json': 'pose_results.json',
                        'crops_folder': 'crops',
                        'jersey_id_result': 'jersey_id_results.json',
                        'final_result': 'final_results.json',
                        'legibility_to_combine': ['raw_legible_resnet18.json', 'raw_legible_resnet34.json', 'raw_legible_vit.json'],
                        'f1_weights': [0.94, 0.94, 0.936]
                    },
                 'val': {
                        'images': 'val/images',
                        'gt': 'val/val_gt.json',
                        'feature_output_folder': 'out/SoccerNetResults/val',
                        'illegible_result': 'illegible_val.json',
                        'legible_result': 'legible_val.json',
                        'soccer_ball_list': 'soccer_ball_val.json',
                        'crops_folder': 'crops_val',
                        'sim_filtered': 'val/main_subject_0.4.json',
                        'gauss_filtered': 'val/main_subject_gauss_th=3.5_r=3.json',
                        'pose_input_json': 'pose_input_val.json',
                        'pose_output_json': 'pose_results_val.json',
                        'jersey_id_result': 'jersey_id_results_validation.json'
                    },
                 'train': {
                     'images': 'train/images',
                     'gt': 'train/train_gt.json',
                     'feature_output_folder': 'out/SoccerNetResults/train',
                     'illegible_result': 'illegible_train.json',
                     'legible_result': 'legible_train.json',
                     'soccer_ball_list': 'soccer_ball_train.json',
                     'sim_filtered': 'train/main_subject_0.4.json',
                     'gauss_filtered': 'train/main_subject_gauss_th=3.5_r=3.json',
                     'pose_input_json': 'pose_input_train.json',
                     'pose_output_json': 'pose_results_train.json',
                     'raw_legible_result': 'train_raw_legible_combined.json',
                     'legibility_to_combine': ['train_raw_legible_resnet18.json', 'train_raw_legible_resnet34.json',
                                               'train_raw_legible_vit.json'],
                     'f1_weights': [0.9307, 0.937, 0.9047]
                 },
                 'challenge': {
                        'images': 'challenge/images',
                        'feature_output_folder': 'out/SoccerNetResults/challenge',
                        'gt': '',
                        'illegible_result': 'challenge_illegible.json',
                        'soccer_ball_list': 'challenge_soccer_ball.json',
                        'sim_filtered': 'challenge/main_subject_0.4.json',
                        'gauss_filtered': 'challenge/main_subject_gauss_th=3.5_r=3.json',
                        'legible_result': 'challenge_legible.json',
                        'pose_input_json': 'challenge_pose_input.json',
                        'pose_output_json': 'challenge_pose_results.json',
                        'crops_folder': 'challenge_crops',
                        'jersey_id_result': 'challenge_jersey_id_results.json',
                        'final_result': 'challenge_final_results.json',
                        'raw_legible_result': 'challenge_raw_legible_vit.json',
                        'legibility_to_combine': ['challenge_raw_legible_resnet18.json', 'challenge_raw_legible_resnet34.json',
                                               'challenge_raw_legible_vit.json'],
                        'f1_weights': [0.94, 0.94, 0.936]
                 },
                 'numbers_data': 'lmdb',
                 #'legibility_model':  "models/resnet18_balanced_soccernet2.pth",
                 #'legibility_model': "experiments/legibility_20240125-175809.pth", # last layer only
                 #'legibility_model':  "experiments/legibility_20230920-172729.pth",
                 #'legibility_model':  "experiments/legibility_20231107-135057.pth",
                 #'legibility_model':  "experiments/legibility_20231219-155728.pth",
                 #'legibility_model': "experiments/legibility_resnet34_20240126-190907.pth",

                 #'legibility_model': "experiments/legibility_resnet18_20240201-172254.pth", #hockey-trained with sam
                 #'legibility_model': "experiments/legibility_vit_20240202-000631.pth", #hockey-trained with sam
                 #'legibility_model': "experiments/legibility_resnet34_20240201-180522.pth", #hockey-trained with sam

                 #'legibility_model': "experiments/legibility_resnet18_20240207-170956.pth", #soccer trained with sam
                 #'legibility_model': "experiments/legibility_resnet18_20240208-121430.pth",

                 'legibility_model': "experiments/legibility_resnet34_20240215-152213.pth",
                 #'legibility_model': "experiments/legibility_vit_20240215-223129.pth",
                 'legibility_model_arch': "resnet34",


                 'legibility_model_url':  "https://drive.google.com/uc?id=1SdIDmlnyuPqqzobapZiohVVAyS61V5VX",
                 'pose_model_url': 'https://drive.google.com/uc?id=1A3ftF118IcxMn_QONndR-8dPWpf7XzdV',
                 #'str_model':'models/parseq_epoch=0-step=582-val_accuracy=91.5826-val_NED=93.9535.ckpt',
                 'str_model': 'models/parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt',
                 #'str_model': 'models/parseq_epoch=3-step=95-val_accuracy=98.7903-val_NED=99.3952.ckpt',
                 #'str_model': 'pretrained=parseq',
                 'str_model_url': "https://drive.google.com/uc?id=1uRln22tlhneVt3P6MePmVxBWSLMsL3bm",
                },
           "Hockey": {
                 'root_dir': 'data/Hockey',
                 'legibility_data': 'legibility_dataset',
                 'numbers_data': 'jersey_number_dataset/jersey_numbers_lmdb',
                 'legibility_model':  'models/resne18_balanced2.pth',
                 'legibility_model_url':  "https://drive.google.com/uc?id=1-wjjfwagysOuSc_wcs4ZurGBUfvcVqO6",
                 #'str_model': 'models/parseq_epoch=3-step=95-val_accuracy=98.7903-val_NED=99.3952.ckpt',
                 #'str_model': 'models/parseq_epoch=0-step=582-val_accuracy=91.5826-val_NED=93.9535.ckpt',
                 'str_model': 'models/parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt',
                 'str_model_url': "https://drive.google.com/uc?id=1FyM31xvSXFRusN0sZH0EWXoHwDfB9WIE"
            }
        }