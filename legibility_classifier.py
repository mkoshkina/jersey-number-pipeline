from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from jersey_number_dataset import JerseyNumberLegibilityDataset, UnlabelledJerseyNumberLegibilityDataset, TrackletLegibilityDataset
from networks import LegibilityClassifier, LegibilitySimpleClassifier, LegibilityClassifier34, LegibilityClassifierTransformer

import time
import copy
import argparse
import os
import configuration as cfg
import time
from tqdm import tqdm
import pandas as pd
import numpy as np

from sam.sam import SAM

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels, _ in dataloaders[phase]:
                #print(f"input and label sizes:{len(inputs), len(labels)}")
                labels = labels.reshape(-1, 1)
                labels = labels.type(torch.FloatTensor)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #print(f"output size is {len(outputs)}")
                    preds = outputs.round()
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def train_model_with_sam(model, criterion, optimizer, num_epochs=25, ):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels, _ in dataloaders[phase]:
                #print(f"input and label sizes:{len(inputs), len(labels)}")
                labels = labels.reshape(-1, 1)
                labels = labels.type(torch.FloatTensor)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    preds = outputs.round()

                    loss = criterion(outputs, labels)  # use this loss for any training statistics
                    if phase == 'train':
                        loss.backward()
                        optimizer.first_step(zero_grad=True)

                        # second forward-backward pass
                        criterion(model(inputs), labels).backward()  # make sure to do a full forward pass
                        optimizer.second_step(zero_grad=True)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def run_full_validation(model, dataloader):
    results = []
    tracks = []
    gt = []

    for inputs, track, label in dataloader:
        # print(f"input and label sizes:{len(inputs), len(labels)}")
        inputs = inputs.to(device)

        # zero the parameter gradients
        torch.set_grad_enabled(False)
        outputs = model_ft(inputs)

        outputs = outputs.float()

        preds = outputs.cpu().detach().numpy()
        flattened_preds = preds.flatten().tolist()
        results += flattened_preds
        tracks += track
        gt += label

    # evaluate tracklet-level accuracy
    unique_tracks = np.unique(np.array(tracks))
    result_dict = {key:[] for key in unique_tracks}
    track_gt = {key:0 for key in unique_tracks}
    for i, result in enumerate(results):
        result_dict[tracks[i]].append(round(result))
        track_gt[tracks[i]] = gt[i]
    correct = 0
    total = 0
    for track in result_dict.keys():
        legible = list(np.nonzero(result_dict[track]))[0]
        if len(legible) == 0 and track_gt[track] == 0:
            correct += 1
        elif len(legible) > 0 and track_gt[track] == 1:
            correct += 1
        total += 1

    return correct/total


def train_model_with_sam_and_full_val(model, criterion, optimizer, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                val_acc = run_full_validation(model, dataloaders['val'])
                print(f'{phase} Acc: {val_acc:.4f}')
                if best_acc < val_acc:
                    best_acc = val_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                continue

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels, _ in dataloaders[phase]:
                #print(f"input and label sizes:{len(inputs), len(labels)}")
                labels = labels.reshape(-1, 1)
                labels = labels.type(torch.FloatTensor)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    preds = outputs.round()

                    loss = criterion(outputs, labels)  # use this loss for any training statistics
                    if phase == 'train':
                        loss.backward()
                        optimizer.first_step(zero_grad=True)

                        # second forward-backward pass
                        criterion(model(inputs), labels).backward()  # make sure to do a full forward pass
                        optimizer.second_step(zero_grad=True)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')


        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test_model(model, subset, result_path=None):
    model.eval()
    running_corrects = 0
    # Iterate over data.
    temp_max = 500
    temp_count = 0
    predictions = []
    gt = []
    raw_predictions = []
    img_names = []
    for inputs, labels, names in tqdm(dataloaders[subset]):
        # print(f"input and label sizes:{len(inputs), len(labels)}")
        temp_count += len(labels)
        inputs = inputs.to(device)
        labels = labels.reshape(-1, 1)
        labels = labels.type(torch.FloatTensor)
        labels = labels.to(device)

        # zero the parameter gradients
        torch.set_grad_enabled(False)
        outputs = model(inputs)
        preds = outputs.round()
        running_corrects += torch.sum(preds == labels.data)
        if subset == 'train' and temp_count >= temp_max:
            break
        gt += labels.data.detach().cpu().numpy().flatten().tolist()
        predictions += preds.detach().cpu().numpy().flatten().tolist()
        raw_predictions += outputs.data.detach().cpu().numpy().flatten().tolist()
        img_names += list(names)

    if subset == 'train':
        epoch_acc = running_corrects.double() / temp_count
    else:
        epoch_acc = running_corrects.double() / dataset_sizes[subset]

    total, TN, TP, FP, FN = 0 ,0, 0, 0, 0
    for i, true_value in enumerate(gt):
        predicted_legible = predictions[i] == 1
        if true_value == 0 and not predicted_legible:
            TN += 1
        elif true_value != 0 and predicted_legible:
            TP += 1
        elif true_value == 0 and predicted_legible:
            FP += 1
        elif true_value != 0 and not predicted_legible:
            FN += 1
        total += 1

    print(f'Correct {TP+TN} out of {total}. Accuracy {100*(TP+TN)/total}%.')
    print(f'TP={TP}, TN={TN}, FP={FP}, FN={FN}')
    Pr = TP / (TP + FP)
    Recall = TP / (TP + FN)
    print(f"Precision={Pr}, Recall={Recall}")
    print(f"F1={2*Pr*Recall/(Pr+Recall)}")

    print(f"Accuracy {subset}:{epoch_acc}")
    print(f"{running_corrects}, {dataset_sizes[subset]}")

    if not result_path is None and len(result_path) > 0:
        with open(result_path, 'w') as f:
            for i, name in enumerate(img_names):
                f.write(f"{name},{round(raw_predictions[i], 2)}\n")

    return epoch_acc


# run inference on a list of files
def run(image_paths, model_path, threshold=0.5, arch='resnet18'):
    # setup data
    dataset = UnlabelledJerseyNumberLegibilityDataset(image_paths, arch=arch)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                                  shuffle=False, num_workers=4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    #load model
    state_dict = torch.load(model_path, map_location=device)
    if arch == 'resnet18':
        model_ft = LegibilityClassifier()
    elif arch == 'vit':
        model_ft = LegibilityClassifierTransformer()
    else:
        model_ft = LegibilityClassifier34()

    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
    model_ft.load_state_dict(state_dict)
    model_ft = model_ft.to(device)
    model_ft.eval()

    # run classifier
    results = []
    for inputs in dataloader:
        # print(f"input and label sizes:{len(inputs), len(labels)}")
        inputs = inputs.to(device)

        # zero the parameter gradients
        torch.set_grad_enabled(False)
        outputs = model_ft(inputs)

        if threshold > 0:
            outputs = (outputs>threshold).float()
        else:
            outputs = outputs.float()
        preds = outputs.cpu().detach().numpy()
        flattened_preds = preds.flatten().tolist()
        results += flattened_preds

    return results


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true', help='fine-tune model by loading public IMAGENET-trained weights')
    parser.add_argument('--sam', action='store_true', help='Use Sharpness-Aware Minimization during training')
    parser.add_argument('--finetune', action='store_true', help='load custom fine-tune weights for further training')
    parser.add_argument('--data', help='data root dir')
    parser.add_argument('--trained_model_path', help='trained model to use for testing or to load for finetuning')
    parser.add_argument('--new_trained_model_path', help='path to save newly trained model')
    parser.add_argument('--arch', choices=['resnet18', 'simple', 'resnet34', 'vit'], default='resnet18', help='what architecture to use')
    parser.add_argument('--full_val_dir', help='to use tracklet instead of images for validation specify val dir')

    args = parser.parse_args()

    annotations_file = '_gt.txt'
    use_full_validation = (not args.full_val_dir is None) and (len(args.full_val_dir) > 0)

    image_dataset_train = JerseyNumberLegibilityDataset(os.path.join(args.data, 'train', 'train' + annotations_file),
                                                        os.path.join(args.data, 'train', 'images'), 'train', isBalanced=True, arch=args.arch)
    if not args.train and not args.finetune:
        image_dataset_test = JerseyNumberLegibilityDataset(os.path.join(args.data, 'test', 'test' + annotations_file),
                                                       os.path.join(args.data, 'test', 'images'), 'test', arch=args.arch)

    dataloader_train = torch.utils.data.DataLoader(image_dataset_train, batch_size=4,
                                                   shuffle=True, num_workers=4)

    if not args.train and not args.finetune:
        dataloader_test = torch.utils.data.DataLoader(image_dataset_test, batch_size=4,
                                                  shuffle=False, num_workers=4)

    # use full validation set during training
    if use_full_validation:
        image_dataset_full_val = TrackletLegibilityDataset(os.path.join(args.full_val_dir, 'val_gt.json'),
                                                          os.path.join(args.full_val_dir, 'images'), arch=args.arch)
        dataloader_full_val = torch.utils.data.DataLoader(image_dataset_full_val, batch_size=4,
                                                     shuffle=False, num_workers=4)
        image_datasets = {'train': image_dataset_train, 'val': image_dataset_full_val}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        dataloaders = {'train': dataloader_train, 'val': dataloader_full_val}

    elif not args.train and not args.finetune:
        image_datasets = {'test': image_dataset_test}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}
        dataloaders = {'test': dataloader_test}
    else:
        image_dataset_val = JerseyNumberLegibilityDataset(os.path.join(args.data, 'val', 'val' + annotations_file),
                                                          os.path.join(args.data, 'val', 'images'), 'val', arch=args.arch)
        dataloader_val = torch.utils.data.DataLoader(image_dataset_val, batch_size=4,
                                                     shuffle=True, num_workers=4)
        image_datasets = {'train': image_dataset_train, 'val': image_dataset_val}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        dataloaders = {'train': dataloader_train, 'val': dataloader_val}

    if args.arch == 'resnet18':
        model_ft = LegibilityClassifier()
    elif args.arch == 'simple':
        model_ft = LegibilitySimpleClassifier()
    elif args.arch == 'vit':
        model_ft = LegibilityClassifierTransformer()
    else:
        model_ft = LegibilityClassifier34()

    if args.train or args.finetune:
        if args.finetune:
            if args.trained_model_path is None or args.trained_model_path == '':
                load_model_path = cfg.dataset["Hockey"]['legibility_model']
            else:
                load_model_path = args.trained_model_path
            # load weights
            state_dict = torch.load(load_model_path, map_location=device)
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            model_ft.load_state_dict(state_dict)

        model_ft = model_ft.to(device)
        criterion = nn.BCELoss()
        if args.sam:
            # Observe that all parameters are being optimized
            base_optimizer = torch.optim.SGD
            optimizer_ft = SAM(model_ft.parameters(), base_optimizer, lr=0.001, momentum=0.9)

            if use_full_validation:
                model_ft = train_model_with_sam_and_full_val(model_ft, criterion, optimizer_ft, num_epochs=10)
            else:
                model_ft = train_model_with_sam(model_ft, criterion, optimizer_ft, num_epochs=10)
        else:
            optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

            # Decay LR by a factor of 0.1 every 7 epochs
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
            model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                   num_epochs=15)

        timestr = time.strftime("%Y%m%d-%H%M%S")
        save_model_path = f"./experiments/legibility_{args.arch}_{timestr}.pth"

        torch.save(model_ft.state_dict(), save_model_path)

    else:
        #load weights
        state_dict = torch.load(args.trained_model_path, map_location=device)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        model_ft.load_state_dict(state_dict)
        model_ft = model_ft.to(device)

        test_model(model_ft, 'test', result_path=args.raw_result_path)