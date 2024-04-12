pose_home = 'pose/ViTPose'
pose_env = 'vitpose'

str_home = 'str/parseq/'
str_env = 'parseq2'
str_platform = 'cu113'

# centroids
reid_env = 'centroids'
reid_script = 'centroid_reid.py'

reid_home = 'reid/'


dataset = {'SoccerNet':
                {'root_dir': './data/SoccerNet',
                 'working_dir': './out/SoccerNetResults',
                 'test': {
                        'images': 'test/images',
                        'gt': 'test/test_gt.json',
                        'feature_output_folder': 'out/SoccerNetResults/test',
                        'illegible_result': 'illegible.json',
                        'soccer_ball_list': 'soccer_ball.json',
                        'sim_filtered': 'test/main_subject_0.4.json',
                        'gauss_filtered': 'test/main_subject_gauss_th=3.5_r=3.json',
                        'legible_result': 'legible.json',
                        'raw_legible_result': 'raw_legible_resnet34.json',
                        'pose_input_json': 'pose_input.json',
                        'pose_output_json': 'pose_results.json',
                        'crops_folder': 'crops',
                        'jersey_id_result': 'jersey_id_results.json',
                        'final_result': 'final_results.json'
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
                     'raw_legible_result': 'train_raw_legible_combined.json'
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
                        'raw_legible_result': 'challenge_raw_legible_vit.json'
                 },
                 'numbers_data': 'lmdb',

                 'legibility_model': "models/legibility_resnet34_soccer_20240215.pth",
                 'legibility_model_arch': "resnet34",

                 'legibility_model_url':  "https://drive.google.com/uc?id=18HAuZbge3z8TSfRiX_FzsnKgiBs-RRNw",
                 'pose_model_url': 'https://drive.google.com/uc?id=1A3ftF118IcxMn_QONndR-8dPWpf7XzdV',
                 'str_model': 'models/parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt',

                 #'str_model': 'pretrained=parseq',
                 'str_model_url': "https://drive.google.com/uc?id=1uRln22tlhneVt3P6MePmVxBWSLMsL3bm",
                },
           "Hockey": {
                 'root_dir': 'data/Hockey',
                 'legibility_data': 'legibility_dataset',
                 'numbers_data': 'jersey_number_dataset/jersey_numbers_lmdb',
                 'legibility_model':  'models/legibility_resnet34_hockey_20240201.pth',
                 'legibility_model_url':  "https://drive.google.com/uc?id=1RfxINtZ_wCNVF8iZsiMYuFOP7KMgqgDp",
                 'str_model': 'models/parseq_epoch=3-step=95-val_accuracy=98.7903-val_NED=99.3952.ckpt',
                 'str_model_url': "https://drive.google.com/uc?id=1FyM31xvSXFRusN0sZH0EWXoHwDfB9WIE",
            }
        }