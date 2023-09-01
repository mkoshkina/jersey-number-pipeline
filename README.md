# General Framework for Jersey Number Recognition in Sports

Image-level detection, localization and recognition (experiments on Hockey dataset):
  - legibility classifier
  - pose extraction
  - scene text recognition for jersey numbers

Tracklet-level detection, localization and recognition (experiments on SoccerNet dataset):
  - occlusion/outlier removal using re-id features and fitting a guassian
  - legibility classifier
  - pose extraction
  - scene text recognition for jersey numbers
  - tracklet prediction consolidation

## Setup:
Code makes use of the following repositories that need to be installed separately following corresponding setup for each repo:

Centroid-Reid:
https://github.com/mikwieczorek/centroids-reid
https://drive.google.com/drive/folders/1NWD2Q0JGasGm9HTcOy4ZqsIqK4-IfknK

ViTPose:
https://github.com/ViTAE-Transformer/ViTPose
Model: https://1drv.ms/u/s!AimBgYV7JjTlgShLMI-kkmvNfF_h?e=dEhGHe

PARSeq:
https://github.com/baudm/parseq

Our code requirements:
pytorch 1.9.0
opencv

## Datasets:
SoccerNet:
https://github.com/SoccerNet/sn-jersey

Hockey: to be released.

## Configuration:
Update configuration.py to point to data and local installations of above repos.
Update path in line 6 of centroid_reid.py to point to your local installation of centroids reid.

Once the models are made available, run:
> python3 main.py SoccerNet test
