# General Framework for Jersey Number Recognition in Sports

Image-level detection, localization and recognition (experiments on Hockey dataset):
  - legibility classifier
  - pose extraction
  - scene text recognition for jersey numbers

Tracklet-level detection, localization and recognition (experiments on SoccerNet dataset):
  - occlusion/outlier removal using re-id features and fitting a guassian
  - legibility classifier
  - pose-guided RoI cropping
  - scene text recognition for jersey numbers
  - tracklet prediction consolidation

## Setup:
Clone current repo.
Create conda environment and install requirements.
Code makes use of the several repositories. Run 
> python3 setup.py 

to automatically clone, setup a separate conda environment for each and fetch models.

### Centroid-Reid:
Repo: https://github.com/mikwieczorek/centroids-reid

Model: https://drive.google.com/drive/folders/1NWD2Q0JGasGm9HTcOy4ZqsIqK4-IfknK

### ViTPose:
Repo: https://github.com/ViTAE-Transformer/ViTPose

Model: https://1drv.ms/u/s!AimBgYV7JjTlgShLMI-kkmvNfF_h?e=dEhGHe

### PARSeq:
Repo: https://github.com/baudm/parseq

### Trained Models:
SoccerNet Legibility classifier: https://drive.google.com/file/d/1SdIDmlnyuPqqzobapZiohVVAyS61V5VX/view?usp=drive_link

SoccerNet-tuned PARSeq: https://drive.google.com/file/d/1uRln22tlhneVt3P6MePmVxBWSLMsL3bm/view?usp=drive_link

### Requirements:
pytorch 1.9.0
opencv

## Datasets:
SoccerNet:
https://github.com/SoccerNet/sn-jersey

Hockey: to be released.

## Configuration:
Update configuration.py as required to set custom path to data or dependencies. 

Once the models are made available, run:
> python3 main.py SoccerNet test
