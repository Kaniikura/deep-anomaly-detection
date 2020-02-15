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
                 transform=None,
                 idx_fold=0,
                 num_fold=1,
                 num_classes=15,
                 onehot_enc = False,
                 csv_filename='csvs/metric_learning.csv',
                 **_):
        self.split = split
        self.idx_fold = idx_fold
        self.num_fold = num_fold
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

        valid_idx = self.idx_fold
        test_idx  = self.num_fold

        if self.split == 'valid':
            df = df[df.Fold == valid_idx ]
        elif self.split == 'test':
            df = df[df.Fold == test_idx]
        elif self.split == 'train' or 'get_embeddings':
            # remove anomaly samples from train data
            df = df[df['Anomaly'] == 0]
            df = df[(df.Fold != valid_idx) & (df.Fold != test_idx)]

        targets = sorted(df.Category.unique())
        label_map = {cat:i for i,cat in enumerate(targets)}
        df['LabelIndex'] = df.Category.apply(lambda x: label_map[x])

        df = df.reset_index()

        return df

    def __getitem__(self, index):
        selected_row = self.df.iloc[index]

        image_path = selected_row['Image']
        image_id = image_path.split('/')[-1]
        image = cv2.imread(image_path).astype(np.float32)
        
        label = selected_row['LabelIndex']
        if self.onehot_enc:
            label_oh = np.zeros(self.num_classes, dtype=np.int)
            label_oh[label_idx] = 1
            label = label_oh

        if self.transform is not None:
            image = self.transform(image)

        return {'image_id': image_id, 'image': image, 'label': label}

    def __len__(self):
        return len(self.df)
    