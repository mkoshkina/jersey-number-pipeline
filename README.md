# General Framework for Jersey Number Recognition in Sports

Image-level detection, localization and recognition (experiments on Hockey dataset):
  - legibility classifier
  - pose-guided RoI cropping
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
```
python3 setup.py 
```

to automatically clone, setup a separate conda environment for each and fetch models.

### Centroid-Reid:
Repo: https://github.com/mikwieczorek/centroids-reid

Model: [https://drive.google.com/drive/folders/1NWD2Q0JGasGm9HTcOy4ZqsIqK4-IfknK](https://drive.google.com/drive/folders/1NWD2Q0JGasGm9HTcOy4ZqsIqK4-IfknK)

### ViTPose:
Repo: https://github.com/ViTAE-Transformer/ViTPose

Model: [https://1drv.ms/u/s!AimBgYV7JjTlgShLMI-kkmvNfF_h?e=dEhGHe](https://1drv.ms/u/s!AimBgYV7JjTlgShLMI-kkmvNfF_h?e=dEhGHe)

### PARSeq:
Repo: [https://github.com/baudm/parseq](https://github.com/baudm/parseq)

### Trained Models:
Hockey Legibility classifier: [https://drive.google.com/file/d/1-wjjfwagysOuSc_wcs4ZurGBUfvcVqO6/view?usp=sharing](https://drive.google.com/file/d/1-wjjfwagysOuSc_wcs4ZurGBUfvcVqO6/view?usp=sharing)

Hockey-tuned PARSeq: [https://drive.google.com/file/d/1FyM31xvSXFRusN0sZH0EWXoHwDfB9WIE/view?usp=sharing]()https://drive.google.com/file/d/1FyM31xvSXFRusN0sZH0EWXoHwDfB9WIE/view?usp=sharing

SoccerNet Legibility classifier: [https://drive.google.com/file/d/1SdIDmlnyuPqqzobapZiohVVAyS61V5VX/view?usp=sharing](https://drive.google.com/file/d/1SdIDmlnyuPqqzobapZiohVVAyS61V5VX/view?usp=sharing)

SoccerNet-tuned PARSeq: [https://drive.google.com/file/d/1uRln22tlhneVt3P6MePmVxBWSLMsL3bm/view?usp=sharing](https://drive.google.com/file/d/1uRln22tlhneVt3P6MePmVxBWSLMsL3bm/view?usp=sharing)
### Requirements:
* pytorch 1.9.0
* opencv

## Datasets:
SoccerNet:
https://github.com/SoccerNet/sn-jersey
Download and save under data subfolder.  Download updated test ground truth file that assigns -1 label to all ball tracks 
that were wrongly labelled with number 1 in originally released data. The file can be downloaded from [here](https://drive.google.com/file/d/1mRnglyMiuuM6CYuzm-ZMFOG72ZeS_8ck/view?usp=sharing) and placed in the test subdirectory of SoccerNet
dataset.

Jersey number crops used to fine-tune STR in LMDB format can be downloaded here: [https://drive.google.com/file/d/1PX8XDF3nNMZAvcjL6M5hurwX78ePAhSs/view?usp=sharing](https://drive.google.com/file/d/1PX8XDF3nNMZAvcjL6M5hurwX78ePAhSs/view?usp=sharing)

Hockey: 
* Download legibility dataset and save under data/Hockey subfolder: [https://drive.google.com/file/d/1Hmm7JDomA_1eOC688OKNCISvLo8emfXW/view?usp=sharing](https://drive.google.com/file/d/1Hmm7JDomA_1eOC688OKNCISvLo8emfXW/view?usp=sharing)
* Download jersey number dataset and save under data/Hockey subfolder: [https://drive.google.com/file/d/1lVoZdOz1RDr6f__MN2irOWf_RHrf-eiD/view?usp=sharing](https://drive.google.com/file/d/1lVoZdOz1RDr6f__MN2irOWf_RHrf-eiD/view?usp=sharing)


## Configuration:
Update configuration.py as required to set custom path to data or dependencies. 

## Inference:
To run the full inference pipeline for SoccerNet:
```
python3 main.py SoccerNet test
```
To run the full inference pipeline for hockey:
```
python3 main.py Hockey test
```
## Train (Hockey)
Train legibility classifier for it:
```
python3 legibility_classifier.py --train --data <new-dataset-directory> --trained_model_path ./experiments/sn_legibility.pth
```

Fine-tune PARSeq STR for hockey number recognition:
```
python3 Hockey train --train_str
```

Trained model will be under str/parseq/outputs

## Train (SoccerNet)
To train legibility classifier and jersey number recognition for SoccerNet, we first generate weakly labelled datasets and then use them to fine-tune.
Weak labels are obtained by using models trained on hockey data.

Generate SoccerNet weakly-labelled legibility data:
```
python3 weak_labels_generation.py --legibility --src <SoccerNet-directory>  --dst <new-dataset-directory>
```

Train legibility classifier for it:
```
python3 legibility_classifier.py --finetune --data <new-dataset-directory> --new_trained_model_path ./experiments/sn_legibility.pth
```

Generate SoccerNet weakly-labelled jersey numbers data:
```
python3 weak_labels_generation.py --numbers --src <SoccerNet-directory>  --dst <new-dataset-directory> --legible_json <legibility-dataset-directory>/legible.json
```

Fine-tune PARSeq on weakly-labelled SoccerNet data:
```
python3 SoccerNet train --train_str
```

Trained model will be under str/parseq/outputs.

## Acknowledgement
We would like to thank authors of the following repositories: 
* [PARSeq](https://github.com/baudm/parseq)
* [Centroid-Reid](https://github.com/mikwieczorek/centroids-reid)
* [ViTPose](https://github.com/ViTAE-Transformer/ViTPose)
* [SoccerNet](https://github.com/SoccerNet/sn-jersey)
* [McGill Hockey Player Tracking Dataset](https://github.com/grant81/hockeyTrackingDataset)
