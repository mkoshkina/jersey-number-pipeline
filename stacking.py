from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from joblib import dump, load
import os
from tqdm import tqdm
import argparse
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

raw_val_results_path = ['experiments/val_legibility_sam_resnet18.txt', 'experiments/val_legibility_sam_resnet34.txt', 'experiments/val_legibility_sam_vit.txt']
gt_val_path = os.path.join('/media/storage/jersey_ids/legibility_dataset_combined', 'val', 'val_gt.txt')
model_path = 'experiments/meta_model_regression.joblib'
raw_test_results_path =  ['experiments/legibility_sam_resnet18.txt', 'experiments/legibility_sam_resnet34.txt', 'experiments/legibility_sam_vit16.txt']
gt_test_path = os.path.join('/media/storage/jersey_ids/legibility_dataset_combined', 'test', 'test_gt.txt')

class MetaDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, ind):
        x = self.X[ind]
        y = self.y[ind]
        return x, y

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(3, 100)
        self.linear2 = nn.Linear(100, 1)
        self.leaky_relu = nn.LeakyReLU(0.1)
    def forward(self, x):
        x = self.leaky_relu(self.linear1(x))
        out = F.sigmoid(self.linear2(x))
        return out

def prepare_data(pred_files, gt_file):
    X = []
    y = []
    preds = []

    for file in pred_files:
        x = pd.read_csv(file, names=['name', 'label'], header=None)
        x = x.sort_values(by=['name'], ascending=True)
        preds.append(x)

    g = pd.read_csv(gt_file, names=['name', 'label'], header=None)
    g = g.sort_values(by=['name'], ascending=True)
    names = g['name'].to_numpy()

    for name in tqdm(names):
        skip = False
        entries = []
        for result in preds:
            query = result[result['name'] == name]['label'].tolist()
            if (len(query) == 0):
                skip = True
                break
            entries.append(query[0])
        if not skip:
            y.append(g[g['name'] == name]['label'].tolist()[0])
            X.append(entries)

    return np.array(X).astype(float), np.array(y).astype(float)

def evaluate(y_pred, y):
    total, TN, TP, FP, FN = 0, 0, 0, 0, 0
    for i, true_value in enumerate(y):
        predicted_legible = round(y_pred[i]) == 1
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nn', action='store_true')
    args = parser.parse_args()

    if not args.nn:
        X, y = prepare_data(raw_val_results_path, gt_val_path)
        model = LinearRegression()
        model.fit(X, y)

        # save model
        dump(model, model_path)

        # test model
        X, y = prepare_data(raw_test_results_path, gt_test_path)

        # evaluate
        y_pred = model.predict(X)
        evaluate(y_pred, y)

    else:
        X, y = prepare_data(raw_val_results_path, gt_val_path)
        train_set = MetaDataset(X, y)
        train_loader = DataLoader(train_set, batch_size=256, shuffle=True)

        model = MLP().to(device)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.BCELoss()

        epochs = 20

        model.train()
        for epoch in range(epochs):
            losses = []
            for batch_num, input_data in enumerate(train_loader):
                optimizer.zero_grad()
                x, y = input_data
                y = y.reshape(-1, 1)
                y = y.type(torch.FloatTensor)
                x = x.to(device).float()
                y = y.to(device)

                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                losses.append(loss.item())

                optimizer.step()

                if batch_num % 40 == 0:
                    print('\tEpoch %d | Batch %d | Loss %6.2f' % (epoch, batch_num, loss.item()))
            print('Epoch %d | Loss %6.2f' % (epoch, sum(losses) / len(losses)))


        X, y = prepare_data(raw_test_results_path, gt_test_path)
        test_set = MetaDataset(X, y)
        test_loader = DataLoader(test_set,  batch_size=256, shuffle=False)
        predictions = []
        gt = []

        for inputs, labels in tqdm(test_loader):
            # print(f"input and label sizes:{len(inputs), len(labels)}")
            inputs = inputs.to(device).float()
            labels = labels.reshape(-1, 1)
            labels = labels.type(torch.FloatTensor)
            labels = labels.to(device)

            # zero the parameter gradients
            torch.set_grad_enabled(False)
            outputs = model(inputs)
            preds = outputs.round()

            gt += labels.data.detach().cpu().numpy().flatten().tolist()
            predictions += preds.detach().cpu().numpy().flatten().tolist()

        evaluate(predictions, y)