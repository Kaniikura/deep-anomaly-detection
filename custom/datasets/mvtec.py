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
class MVTecGANDataset(Dataset):
    def __init__(self,
                 data_dir,
                 split,
                 transform=None,
                 idx_fold=0,
                 num_fold=1,
                 csv_filename='train.ver0.csv',
                 submission_filename='sample_submission.csv',
                 **_):
        self.split = split
        self.idx_fold = idx_fold
        self.num_fold = num_fold
        self.transform = transform
        self.data_dir = data_dir
        self.csv_filename = csv_filename
        self.submission_filename = submission_filename

        self.df_examples, self.image_paths = self._load_examples()

    def _load_examples(self):
        csv_path = os.path.join(self.data_dir, self.csv_filename)

        df_examples = pd.read_csv(csv_path)

        df_examples = df_examples.fillna('')

        train_idx = self.idx_fold
        test_idx  = self.num_fold

        if self.split == 'test':
            df_examples = df_examples[df_examples.Fold == test_idx ]
        elif self.split == 'train':
            df_examples = df_examples[df_examples.Fold == train_idx]

        df_examples = df_examples.set_index('Image')
        image_paths = list(df_examples.index.unique())

        return df_examples, image_paths

    def __getitem__(self, index):
        image_suf_path = self.image_paths[index]
        image_id = os.path.split(image_suf_path)[-1]
        image_path = os.path.join(self.data_dir, image_suf_path)

        image = cv2.imread(image_path)
        
        label = self.df_examples.loc[image_suf_path]['Is_Anomaly']

        if self.transform is not None:
            image = self.transform(image)

        return {'image_id': image_id, 'image': image, 'label': label}

    def __len__(self):
        return len(self.image_paths)
    