pose_home = 'pose/ViTPose'
pose_env = 'vitpose2'
# TODO: include download instructions

str_home = 'str/parseq/'
str_env = 'parseq2'
str_platform = 'cu113'

# TODO: figure out better way to handle this
reid_env = 'centroids'
reid_home = 'reid/'

dataset = {'SoccerNet':
                {'root_dir': './data/SoccerNet',
                 'working_dir': './out/SoccerNetResults',
                 'test': {
                        'images': 'test/images',
                        'gt': 'test/test_gt_updated.json',
                        'feature_output_folder': 'out/SoccerNetResults/test',
                    },
                 'legibility_model':  "models/resnet18_balanced_soccernet2.pth",
                 'legibility_model_url':  "https://drive.google.com/uc?id=1SdIDmlnyuPqqzobapZiohVVAyS61V5VX",
                 'illegible_result': 'illegible.json',
                 'sim_filtered': 'test/main_subject_0.4.json',
                 'gauss_filtered': 'test/main_subject_gauss_th=3.5_r=3.json',
                 'legible_result': 'legible.json',
                 'pose_input_json': 'pose_input.json',
                 'pose_output_json': 'pose_results.json',
                 'pose_model_url': 'https://drive.google.com/uc?id=1A3ftF118IcxMn_QONndR-8dPWpf7XzdV',
                 'crops_folder': 'crops',
                 'str_model': 'models/parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt',
                 'str_model_url': "https://drive.google.com/uc?id=1uRln22tlhneVt3P6MePmVxBWSLMsL3bm",
                 'jersey_id_result': 'jersey_id_results.json',
                 'final_result': 'final_results.json'
                },
           "Hockey": {
                 'legibility_model':  'models/resne18_balanced2.pth',
                 'legibility_model_url':  "https://drive.google.com/uc?id=1-wjjfwagysOuSc_wcs4ZurGBUfvcVqO6",
                 'str_model': 'models/epoch=3-step=95-val_accuracy=98.7903-val_NED=99.3952.ckpt',
                 'str_model_url': "https://drive.google.com/uc?id=1FyM31xvSXFRusN0sZH0EWXoHwDfB9WIE"
            }
        }