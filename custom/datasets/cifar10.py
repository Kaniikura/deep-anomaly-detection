from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import numpy as np
import pandas as pd
import cv2
from torch.utils.data.dataset import Dataset

import dlcommon

@dlcommon.DATASETS.register
class CifarMetricDataset(Dataset):
    def __init__(self,
                 data_dir,
                 split,
                 fold_idx,
                 num_folds,
                 transform=None,
                 num_classes=10,
                 anomaly_classes=['bird', 'car', 'dog'],
                 onehot_enc = False,
                 csv_filename='csvs/metric_learning.csv',
                 **_):
        self.split = split
        self.fold_idx = fold_idx # hold-out
        self.num_folds = num_folds
        self.transform = transform
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.anomaly_classes = anomaly_classes
        self.onehot_enc = onehot_enc
        self.csv_filename = csv_filename

        self.df = self._load_examples()

    def _load_examples(self):
        csv_path = os.path.join(self.data_dir, self.csv_filename)

        df = pd.read_csv(csv_path)

        df = df.fillna('')
        df['OrignalIndex'] = df.index

        targets = sorted(df.Category.unique())

        if self.split in ['train' , 'get_embeddings']:
            folds = [i for i in range(self.num_folds) if i != self.fold_idx]
            df = df[df.Fold.isin(folds)]
        else:
            folds = [self.fold_idx]
            df = df[df.Fold.isin(folds)]

        if self.split in ['train', 'validation'] and (self.anomaly_classes is not None):
            # remove anomaly samples in training
            df = df[~df.Category.isin(self.anomaly_classes)]

        df = df.reset_index()

        return df

    def __getitem__(self, index):
        selected_row = self.df.iloc[index]

        image_path = selected_row['Image']
        image_id = image_path.split('/')[-1]
        image = cv2.imread(image_path)
        
        label = selected_row['LabelIndex']
        if self.onehot_enc:
            label_oh = np.zeros(self.num_classes, dtype=np.int)
            label_oh[label_idx] = 1
            label = label_oh

        if self.anomaly_classes is not None:
            is_anomaly = int(selected_row['Category'] in self.anomaly_classes)
        else:
            is_anomaly = 0

        org_index = selected_row['OrignalIndex']

        if self.transform is not None:
            image = self.transform(image)

        return {'image_id': image_id, 'image': image, 'label': label,
                'is_anomaly': is_anomaly, 'index':org_index}

    def __len__(self):
        return len(self.df)

@dlcommon.DATASETS.register
class CifarUnsvDataset(Dataset):
    def __init__(self,
                 data_dir,
                 split,
                 fold_idx,
                 num_folds,
                 anomaly_classes=['bird', 'car', 'dog'],
                 transform=None,
                 csv_filename='csvs/unsv_learning.csv',
                 **_):
        self.split = split
        self.fold_idx = fold_idx # hold-out
        self.num_folds = num_folds
        self.anomaly_classes = anomaly_classes
        self.transform = transform
        self.data_dir = data_dir
        self.csv_filename = csv_filename

        self.df = self._load_examples()

    def _load_examples(self):
        csv_path = os.path.join(self.data_dir, self.csv_filename)

        df = pd.read_csv(csv_path)

        df = df.fillna('')
        df['OrignalIndex'] = df.index

        df = df[df['Category']==self.target]
        if self.split == 'train':
            folds = [i for i in range(self.num_folds) if i != self.fold_idx]
            df = df[df.Fold.isin(folds)]
        else:
            folds = [self.fold_idx]
            df = df[df.Fold.isin(folds)]

        if self.split=='train' and (self.anomaly_classes is not None):
            # remove anomaly samples in training
            df = df[~df.Category.isin(self.anomaly_classes)]

        df = df.reset_index()

        return df

    def __getitem__(self, index):
        selected_row = self.df.iloc[index]

        image_path = selected_row['Image']
        image_id = image_path.split('/')[-1]
        image = cv2.imread(image_path)

        org_index = selected_row['OrignalIndex']

        is_anomaly = int(selected_row['Category'] in self.anomaly_classes)

        if self.transform is not None:
            image = self.transform(image)

        return {'image_id': image_id, 'image': image,
                'is_anomaly': is_anomaly, 'index':org_index}

    def __len__(self):
        return len(self.df)
    