from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from jersey_number_dataset import JerseyNumberLegibilityDataset, UnlabelledJerseyNumberLegibilityDataset
from networks import LegibilityClassifier

import time
import copy
import argparse
import os
import numpy as np


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
            for inputs, labels in dataloaders[phase]:
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

def test_model(model, subset):
    model.eval()
    running_corrects = 0
    # Iterate over data.
    temp_max = 500
    temp_count = 0
    for inputs, labels in dataloaders[subset]:
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
        #print(preds, labels.data)
        running_corrects += torch.sum(preds == labels.data)
        if subset == 'train' and temp_count >= temp_max:
            break

    if subset == 'train':
        epoch_acc = running_corrects.double() / temp_count
    else:
        epoch_acc = running_corrects.double() / dataset_sizes[subset]

    print(f"Accuracy {subset}:{epoch_acc}")
    print(f"{running_corrects}, {dataset_sizes[subset]}")
    return epoch_acc


# run inference on a list of files
def run(image_paths, model_path, threshold=0.5):
    # setup data
    dataset = UnlabelledJerseyNumberLegibilityDataset(image_paths)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                                  shuffle=False, num_workers=4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    #load model
    state_dict = torch.load(model_path, map_location=device)
    model_ft = LegibilityClassifier()
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

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--data', help='data root dir')
    parser.add_argument('--trained_model_path', help='trained model to use for testing or to load for finetuning')
    parser.add_argument('--new_trained_model_path', help='path to save newly trained model')

    args = parser.parse_args()

    annotations_file = '_gt.txt'


    image_dataset_train = JerseyNumberLegibilityDataset(os.path.join(args.data, 'train' + annotations_file),
                                                        os.path.join(args.data, 'images'), 'train', isBalanced=False)
    image_dataset_val = JerseyNumberLegibilityDataset(os.path.join(args.data, 'val' + annotations_file),
                                                      os.path.join(args.data, 'images'), 'val')
    image_dataset_test = JerseyNumberLegibilityDataset(os.path.join(args.data, 'val' + annotations_file),
                                                       os.path.join(args.data, 'images'), 'test')

    dataloader_train = torch.utils.data.DataLoader(image_dataset_train, batch_size=4,
                                                   shuffle=True, num_workers=4)
    dataloader_val = torch.utils.data.DataLoader(image_dataset_val, batch_size=4,
                                                 shuffle=True, num_workers=4)
    dataloader_test = torch.utils.data.DataLoader(image_dataset_test, batch_size=4,
                                                  shuffle=False, num_workers=4)

    image_datasets = {'train': image_dataset_train, 'val': image_dataset_val, 'test': image_dataset_test}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    dataloaders = {'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test}


    model_ft = LegibilityClassifier()
    # create the model based on ResNet18 and train from pretrained version
    if args.train:
        model_ft = model_ft.to(device)
        criterion = nn.BCELoss()

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                               num_epochs=10)
        torch.save(model_ft.state_dict(), args.trained_model_path)

    elif args.finetune:
        #load weights
        state_dict = torch.load(args.trained_model_path, map_location=device)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        model_ft.load_state_dict(state_dict)
        model_ft = model_ft.to(device)

        criterion = nn.BCELoss()

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                               num_epochs=10)
        torch.save(model_ft.state_dict(), args.new_trained_model_path)

    else:
        #load weights
        state_dict = torch.load(args.trained_model_path, map_location=device)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        model_ft.load_state_dict(state_dict)
        model_ft = model_ft.to(device)

        test_model(model_ft, 'test')
