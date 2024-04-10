import torch.nn as nn
from torch.utils import data
import json
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler

TRAIN_X_FILE = "/media/storage/jersey_ids/SoccerNetResults/jersey_id_results_validation.json"
TRAIN_Y_FILE = "/media/storage/jersey_ids/SoccerNet/val/val_gt.json"
MODEL_PATH = "experiments/lstm.pth"
HIDDEN_SIZE = 10
OUTPUT_SIZE = 23

#### CLASSES #######
class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=22, hidden_size=HIDDEN_SIZE, num_layers=2, batch_first=True, bidirectional=False)
        #self.rnn = nn.RNN(input_size=22, hidden_size=HIDDEN_SIZE, num_layers=2, batch_first=True)
        self.linear1 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear2 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x, _ = self.lstm(x)
        #last_h = x[:,-1,:]
        #last_h = x.view(:, -1, :)
        # shape of sequence is: batch_size, seq_size, dim
        sequence = x.swapaxes(0, 1)
        # shape of sequence is: seq_size, batch_size, dim
        x = sequence[-1]
        # shape of sequence is: batch_size, dim (ie last seq is taken)
        x = self.leaky_relu(x)
        x = self.linear1(x)
        x = self.leaky_relu(x)
        x = self.linear2(x)
        #x = self.leaky_relu(x)
        return x

class EncodedNumberDataset(data.Dataset):
    def __init__(self, x, lens, y=None):
        self.x = x
        self.lens = lens
        self.y = y

    def __len__(self):
        return len(self.lens)

    def __getitem__(self, idx):
        if self.y is None:
            return torch.tensor(self.x[idx], dtype=torch.float32), self.lens[idx]
            #return self.x[idx]
        #return self.x[idx], self.y[idx]
        return torch.tensor(self.x[idx], dtype=torch.float32), self.lens[idx], self.y[idx]


class Collator(object):
    def __init__(self, test=False):
        self.test = test

    def __call__(self, batch):
        global MAX_LEN

        if self.test:
            x, lens = zip(*batch)
            #x = zip(*batch)
        else:
            x, lens, target = zip(*batch)
            #x, target = zip(*batch)

        lens = np.array(lens)
        max_length = np.max(lens)
        #torch.tensor(x, dtype=torch.float32)
        tensor_x = pad_sequence(x, batch_first=True)
        tensor_x =tensor_x.cuda()
        if self.test:
            return tensor_x

        return tensor_x, torch.tensor(target, dtype=torch.float32).cuda()


#########################################

############ UTILS ######################
token_list = 'E0123456789'
def encode_label(label):
    label = int(label)
    vector = [0 for x in range(23)]
    if label == -1:
        vector[-1] = 1
        return vector
    if len(str(label)) > 1:
        tens = str(label//10)
        ones = str(label % 10)
        vector[token_list.index(tens)] = 1
        vector[token_list.index(ones)+11] = 1
    else:
        ones = str(label % 10)
        vector[token_list.index(ones)] = 1
        vector[11] = 1
    return vector

def decode_label(vector):
    if vector[22] == 1:
        return -1
    tens_i = np.argmax(vector[:11])
    ones_i = np.argmax(vector[11:])
    if tens_i == token_list.index('E'):
        return -1
    tens = int(token_list[tens_i])
    if ones_i == token_list.index('E'):
        return tens
    else:
        return tens*10 + int(token_list[ones_i])

MAX_LENGTH = 25
def parse_data(train_x, train_y):
    x = []
    y = []
    x_lens = []
    dict = {}
    # re-organize predictions into dictionary of tracks
    for key in train_x.keys():
        name = key.split('.')[0]
        parts = name.split('_')
        track = int(parts[0])
        img_indx = int(parts[1])
        # make one big array for both digits
        raw = train_x[key]['raw']
        if len(raw) > 1:
            entry = raw[0] + raw[1]
        if not track in dict.keys():
            dict[track] = {}
        dict[track][img_indx] = entry

    # count = 0
    # for t in train_y.keys():
    #     if not int(t) in dict.keys():
    #         count += 1
    #         print(f"no images for {t}, label {train_y[t]}")
    # print(f"missing {count}")

    for key in dict.keys():
        if not str(key) in train_y.keys():
            print(f"Missing gt for {key}")
            continue
        img_keys = list(dict[key].keys())
        img_keys.sort()
        series = []
        for ik in img_keys:
            if len(series) == MAX_LENGTH:
                x.append(series)
                y.append(encode_label(train_y[str(key)]))
                x_lens.append(len(series))
                series = []
            series.append(dict[key][ik])
        x.append(series)
        encoded = encode_label(train_y[str(key)])
        assert train_y[str(key)] == decode_label(encoded), f'{train_y[str(key)]},{decode_label(encoded)}'
        y.append(encoded)
        x_lens.append(len(series))

    return x, x_lens, y

def test_model(model, dataloader):
    for inputs, labels in dataloader:
        outputs = torch.sigmoid(model(inputs.cuda()))
        preds = list(torch.round(outputs).detach().cpu())
        pred = [decode_label(x) for x in preds]
        labels = list(labels.detach().cpu())
        label = [decode_label(y) for y in labels]
        print(pred, label)


def train_model(model, dataloader, train_dataset_size, num_epochs=60):

    criterion = nn.BCEWithLogitsLoss()#nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloader:
            # zero the parameter gradients
            optimizer.zero_grad()
            Before = list(model.parameters())[0].clone()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                #print(f"output size is {len(outputs)}")
                preds = torch.round(torch.sigmoid(outputs))
                #print(preds, labels)
                loss = criterion(outputs, labels)
                #print(loss)

                loss.backward()
                optimizer.step()
                After = list(model.parameters())[0].clone()
                print(torch.equal(Before.data, After.data))

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            #print(inputs[0], outputs[0], labels.data[1], preds[1])

        scheduler.step()

        epoch_loss = running_loss / train_dataset_size
        epoch_acc = running_corrects.double() / train_dataset_size

        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return model


if __name__ == '__main__':
    # read prediction file and labels and make a dataset
    with open(TRAIN_X_FILE, 'r') as f:
        train_x = json.load(f)
    with open(TRAIN_Y_FILE, 'r') as f:
        train_y = json.load(f)

    x, x_len, y = parse_data(train_x, train_y)
    train_collate = Collator()
    train_dataset = EncodedNumberDataset(x, x_len, y)
    train_dataset_size = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=train_collate)

    model = LSTMClassifier()
    model.cuda()

    model = train_model(model, train_loader, train_dataset_size)
    # save model
    #torch.save(model.state_dict(), MODEL_PATH)

    # state_dict = torch.load(MODEL_PATH)
    # if hasattr(state_dict, '_metadata'):
    #     del state_dict._metadata
    # model.load_state_dict(state_dict)
    # model.cuda()

    # test model
    test_model(model, train_loader)
