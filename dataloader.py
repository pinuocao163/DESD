import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
import numpy as np


class IEMOCAPRobertaCometDataset(Dataset):

    def __init__(self, path=None, split=None):

        self.speakers, self.labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        self.sentences, self.trainIds, self.testIds, self.validIds \
            = pickle.load(open(path, 'rb'), encoding='latin1')

        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'train-valid':
            self.keys = [x for x in self.trainIds] + [x for x in self.validIds]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(np.array(self.roberta1[vid])), \
               torch.FloatTensor([[1, 0] if x == 'M' else [0, 1] for x in self.speakers[vid]]), \
               torch.FloatTensor([1] * len(self.labels[vid])), \
               torch.LongTensor(self.labels[vid]), \
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 2 else pad_sequence(dat[i], True) if i < 4 else dat[i].tolist() for i in dat]


class MELDRobertaCometDataset(Dataset):

    def __init__(self, path=None, split=None, classify='emotion'):


        self.speakers, self.emotion_labels, self.sentiment_labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        self.sentences, self.trainIds, self.testIds, self.validIds \
            = pickle.load(open(path, 'rb'), encoding='latin1')

        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]

        if classify == 'emotion':
            self.labels = self.emotion_labels
        else:
            self.labels = self.sentiment_labels

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(np.array(self.roberta1[vid])), \
               torch.FloatTensor(self.speakers[vid]), \
               torch.FloatTensor([1] * len(self.labels[vid])), \
               torch.LongTensor(self.labels[vid]), \
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 2 else pad_sequence(dat[i], True) if i < 4 else dat[i].tolist() for i in dat]


class DailyDialogRobertaCometDataset(Dataset):

    def __init__(self, path=None, split=None, classify='emotion'):

        self.speakers, self.emotion_labels, \
            self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
            self.sentences, self.trainIds, self.testIds, self.validIds \
            = pickle.load(open(path, 'rb'), encoding='latin1')


        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]

        if classify == 'emotion':
            self.labels = self.emotion_labels
        else:
            self.labels = self.sentiment_labels

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(np.array(self.roberta1[vid])), \
            torch.FloatTensor([[1, 0] if x == '0' else [0, 1] for x in self.speakers[vid]]), \
            torch.FloatTensor([1] * len(self.labels[vid])), \
            torch.LongTensor(self.labels[vid]), \
            vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 2 else pad_sequence(dat[i], True) if i < 4 else dat[i].tolist() for i in
                dat]


class EmoryNLPRobertaCometDataset(Dataset):

    def __init__(self, path=None, split=None, classify='emotion'):

        self.speakers, self.emotion_labels, \
            self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
            self.sentences, self.trainIds, self.testIds, self.validIds \
            = pickle.load(open(path, 'rb'), encoding='latin1')

        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]

        if classify == 'emotion':
            self.labels = self.emotion_labels
        else:
            self.labels = self.sentiment_labels

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(np.array(self.roberta1[vid])), \
            torch.FloatTensor(self.speakers[vid]), \
            torch.FloatTensor([1] * len(self.labels[vid])), \
            torch.LongTensor(self.labels[vid]), \
            vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 2 else pad_sequence(dat[i], True) if i < 4 else dat[i].tolist() for i in
                dat]


