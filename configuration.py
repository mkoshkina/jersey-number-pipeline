ViTPoseHome = '{UPDATE}/ViTPose'
vitPoseEnv = 'vitpose'

ParSeqHome = '{UPDATE}/parseq/'
parseqEnv = 'parseq'
parseqCheckpoint = 'models/parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt'


centroidEnv = 'centroidReid'

dataset = {'SoccerNet':
                {'root_dir': '{UPDATE}/SoccerNet',
                 'working_dir': '{UPDATE}/SoccerNetResults',
                 'test': {
                        'images': 'test/images',
                        'gt': 'test/test_gt_updated.json',
                        'feature_output_folder': '{UPDATE}/SoccerNetResults/test',
                    },
                 'legibility_model':  "models/resnet18_balanced_soccernet2.pth",
                 'illegible_result': 'illegible.json',
                 'sim_filtered': 'test/main_subject_0.4.json',
                 'gauss_filtered': 'test/main_subject_gauss_th=3.5_r=3.json',
                 'legible_result': 'legible.json',
                 'pose_input_json': 'pose_input.json',
                 'pose_output_json': 'pose_results.json',
                 'crops_folder': 'crops',
                 'jersey_id_result': 'jersey_id_results.json',
                 'final_result': 'final_results.json'
                }
            }