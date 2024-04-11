import os
import configuration as cfg
import json
import urllib.request
import gdown
import argparse


###### common setup utils ##############

def make_conda_env(env_name, libs=""):
    os.system(f"conda create -n {env_name} -y "+libs)

def activate_conda_env(env_name):
    os.system(f"conda activate {env_name}")

def deactivate_conda_env(env_name):
    os.system(f"conda deactivate")

def conda_pyrun(env_name, exec_file, args):
    os.system(f"conda run -n {env_name} --live-stream python3 \"{exec_file}\" '{json.dumps(dict(vars(args)))}'")


def get_conda_envs():
    stream = os.popen("conda env list")
    output = stream.read()
    a = output.split()
    a.remove("*")
    a.remove("#")
    a.remove("#")
    a.remove("conda")
    a.remove("environments:")
    return a[::2]
###########################################


def setup_reid(root):
    env_name  = cfg.reid_env
    repo_name = "centroids-reid"
    src_url   = "https://github.com/mikwieczorek/centroids-reid.git"
    rep_path  = "./reid"

    if not repo_name in os.listdir(rep_path):
        # clone source repo
        os.system(f"git clone --recurse-submodules {src_url} {os.path.join(rep_path, repo_name)}")

        # create the models folder inside repo, weights will be added to that folder later on
        models_folder_path = os.path.join(rep_path, repo_name, "models")
        os.system(f"mkdir {models_folder_path}")

        url = "https://drive.google.com/uc?export=download&id=1w9yzdP_5oJppGIM4gs3cETyLujanoHK8&confirm=t&uuid=fed3cb8a-1fad-40bd-8922-c41ededc93ae&at=ALgDtsxiC0WTza4g47gqC5VPyWg4:1679009047787"
        save_path = os.path.join(models_folder_path, "dukemtmcreid_resnet50_256_128_epoch_120.ckpt")
        urllib.request.urlretrieve(url, save_path)

        url = "https://drive.google.com/uc?export=download&id=1ZFywKEytpyNocUQd2APh2XqTe8X0HMom&confirm=t&uuid=450bb8b7-b3d0-4465-b0c9-bb6f066b205e&at=ALgDtswylGfYgY71u8ZmWx4CfhJX:1679008688985"
        save_path = os.path.join(models_folder_path, "market1501_resnet50_256_128_epoch_120.ckpt")
        urllib.request.urlretrieve(url, save_path)

    if not env_name in get_conda_envs():
        make_conda_env(env_name, libs="python=3.8")
        cwd = os.getcwd()
        os.chdir(os.path.join(rep_path, repo_name))
        os.system(f"conda run --live-stream -n {env_name} conda install --name {env_name} pip")
        os.system(f"conda run --live-stream -n {env_name} pip install -r requirements.txt")

        os.chdir(cwd)

# clone and install vitpose
# download the model
def setup_pose(root):
    env_name  = cfg.pose_env
    repo_name = "ViTPose"
    src_url   = "https://github.com/ViTAE-Transformer/ViTPose.git"
    rep_path  = "./pose"
    os.chdir(root)

    if not repo_name in os.listdir(rep_path):
       # clone source repo
        os.chdir(root)
        os.system(f"git clone --recurse-submodules {src_url} {os.path.join(rep_path,repo_name)}")

    os.chdir(root)
    if not env_name in get_conda_envs():
        make_conda_env(env_name, libs="python=3.8")

        os.system(f"conda run --live-stream -n {env_name} conda install --name {env_name} pip")
        os.system(f"conda run --live-stream -n {env_name} pip install  mmcv-full==1.4.8 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html")

        os.chdir(os.path.join(root, rep_path, "ViTPose"))
        os.system(f"conda run --live-stream -n {env_name} pip install -v -e .")
        os.system(f"conda run --live-stream -n {env_name} pip install timm==0.4.9 einops")


# clone and install str
# download the model
def setup_str(root):
    env_name  = cfg.str_env
    repo_name = "parseq"
    src_url   = "https://github.com/baudm/parseq.git"
    rep_path  = "./str"
    os.chdir(root)

    if not repo_name in os.listdir(rep_path):
        # clone source repo
        os.system(f"git clone --recurse-submodules {src_url} {os.path.join(rep_path, repo_name)}")

    os.chdir(os.path.join(rep_path, repo_name))

    if not env_name in get_conda_envs():
        make_conda_env(env_name, libs="python=3.9")
        os.system(f"make torch-cu117")
        os.system(f"conda run --live-stream -n {env_name} conda install --name {env_name} pip")
        os.system(f"conda run --live-stream -n {env_name} pip install -r requirements/core.cu117.txt -e .[train,test]")

    os.chdir(root)

def download_models_common(root_dir):
    repo_name = "ViTPose"
    rep_path = "./pose"

    url = cfg.dataset['SoccerNet']['pose_model_url']
    models_folder_path = os.path.join(rep_path, repo_name, "checkpoints")
    if not os.path.exists(models_folder_path):
        os.system(f"mkdir {models_folder_path}")
    save_path = os.path.join(rep_path, "ViTPose", "checkpoints", "vitpose-h.pth")
    if not os.path.isfile(save_path):
        gdown.download(url, save_path)

def download_models(root_dir, dataset):
    # download and save fine-tuned model
    save_path = os.path.join(root_dir, cfg.dataset[dataset]['str_model'])
    if not os.path.isfile(save_path):
        source_url = cfg.dataset[dataset]['str_model_url']
        gdown.download(source_url, save_path)

    save_path = os.path.join(root_dir, cfg.dataset[dataset]['legibility_model'])
    if not os.path.isfile(save_path):
        source_url = cfg.dataset[dataset]['legibility_model_url']
        gdown.download(source_url, save_path)

def setup_sam(root_dir):
    os.chdir(root_dir)
    repo_name = 'sam2'
    src_url = 'https://github.com/davda54/sam'

    if not repo_name in os.listdir(root_dir):
        # clone source repo
        os.system(f"git clone --recurse-submodules {src_url} {os.path.join(root_dir, repo_name)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', default='all', help="Options: all, SoccerNet, Hockey")

    args = parser.parse_args()

    root_dir = os.getcwd()

    # common for both datasets
    setup_sam(root_dir)
    setup_pose(root_dir)
    download_models_common(root_dir)
    setup_str(root_dir)

    #SoccerNet only
    if not args.dataset == 'Hockey':
        setup_reid(root_dir)
        download_models(root_dir, 'SoccerNet')

    if not args.dataset == 'SoccerNet':
        download_models(root_dir, 'Hockey')
