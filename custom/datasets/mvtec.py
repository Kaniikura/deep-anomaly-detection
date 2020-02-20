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
class MetricDataset(Dataset):
    def __init__(self,
                 data_dir,
                 split,
                 folds,
                 transform=None,
                 num_classes=15,
                 onehot_enc = False,
                 csv_filename='csvs/metric_learning.csv',
                 **_):
        self.split = split
        if isinstance(folds, str):
            folds = [int(i) for i in folds.split(',') if i!='']
        self.folds = folds
        self.transform = transform
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.onehot_enc = onehot_enc
        self.csv_filename = csv_filename

        self.df = self._load_examples()

    def _load_examples(self):
        csv_path = os.path.join(self.data_dir, self.csv_filename)

        df = pd.read_csv(csv_path)

        df = df.fillna('')
        df['OrignalIndex'] = df.index
        df = df[df.Fold.isin(self.folds)]
        if self.split in ['train' , 'get_embeddings']:
            # remove anomaly samples from train data
            df = df[df['Anomaly'] == 0]
            

        targets = sorted(df.Category.unique())
        label_map = {cat:i for i,cat in enumerate(targets)}
        df['LabelIndex'] = df.Category.apply(lambda x: label_map[x])

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

        is_anomaly = selected_row['Anomaly']
        org_index = selected_row['OrignalIndex']

        if self.transform is not None:
            image = self.transform(image)

        return {'image_id': image_id, 'image': image, 'label': label, 
                'is_anomaly': is_anomaly, 'index':org_index}

    def __len__(self):
        return len(self.df)

@dlcommon.DATASETS.register
class UnsvDataset(Dataset):
    def __init__(self,
                 data_dir,
                 split,
                 folds,
                 target='bottle',
                 transform=None,
                 csv_filename='csvs/unsv_learning.csv',
                 **_):
        self.split = split
        if isinstance(folds, str):
            folds = [int(i) for i in folds.split(',') if i!='']
        self.folds = folds
        self.target = target
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
        df = df[df['Fold'].isin(self.folds)]

        df = df.reset_index()

        return df

    def __getitem__(self, index):
        selected_row = self.df.iloc[index]

        image_path = selected_row['Image']
        image_id = image_path.split('/')[-1]
        image = cv2.imread(image_path)

        is_anomaly = selected_row['Anomaly']
        org_index = selected_row['OrignalIndex']

        if self.transform is not None:
            image = self.transform(image)

        return {'image_id': image_id, 'image': image,
                'is_anomaly': is_anomaly, 'index':org_index}

    def __len__(self):
        return len(self.df)
    