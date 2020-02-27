import numpy as np
import os
import pandas as pd
from pathlib import Path
from utils import *

CIFAR10_LABELMAP = {
    'plane' : 0,
    'car' : 1,
    'bird' : 2,
    'cat' : 3,
    'deer' : 4,
    'dog' : 5,
    'frog': 6,
    'horse' : 7,
    'ship' : 8,
    'truck' : 9,
}
def make_metric_learning_df(data_dir, data_type):
    def _make_df_split(data_dir, data_type, split):
        img_dir = data_dir/data_type/'images'/split
        assert img_dir.is_dir()

        list_of_dicts = []
        sub_dirs = [d for d in img_dir.glob('[A-Za-z0-9]*') if d.is_dir()]
        for sd in sub_dirs:
            if data_type == 'mvtec_ad':
                for ssd in sd.glob('[A-Za-z0-9]*'):
                    assert ssd.is_dir()
                    for img_path in ssd.glob('*.png'):
                        cat = sd.name
                        st = ssd.name
                        anml = 0 if st=='good' else 1
                        list_of_dicts.append({'Image':img_path, 'Status':st,
                                              'Category':cat ,
                                              'Anomaly':anml,
                                              'CatAnml': f'{cat}_{anml}',
                                              'OrgSplit':split})

            else:
                for img_path in sd.glob('*.png'):
                    list_of_dicts.append({'Image':img_path, 'Category':sd.name,
                                          'OrgSplit':split})
        _df = pd.DataFrame(list_of_dicts)

        return _df
    
    df_train = _make_df_split(data_dir, data_type, 'train')
    df_test  = _make_df_split(data_dir, data_type, 'test')
    df = pd.concat([df_train,df_test], ignore_index=True)
    
    return df

def create_csvs(data_dir, data_type):
    if data_type == 'mvtec_ad':
        target = 'CatAnml'
        # metlic learning should be evaluated with cross-validation for MVTec-AD, 
        # cause original train data has no anomaly instances
        df = make_metric_learning_df(data_dir, data_type)
        metric_learning_df = take_stratified_split(df, target, n_splits=10)
        del metric_learning_df[target]
        # take train valid split for unsupervised learning (train/valid data consists of normal data)
        test_idx = df[df['OrgSplit']=='test'].index
        unsv_learning_df = take_stratified_split(df, target, n_splits=1, valid_size=0.3, test_idx=test_idx)
    else:
        target = 'Category'
        df =  make_metric_learning_df(data_dir, data_type)
        df.loc[df['OrgSplit']=='train','Fold'] = 0
        df.loc[df['OrgSplit']=='test' ,'Fold'] = 1
        df.Fold = df.Fold.astype(int)
        df = df.reset_index()
        if data_type == 'cifar10':
            df['LabelIndex'] = df['Category'].apply(lambda x: CIFAR10_LABELMAP[x])
    
        metric_learning_df = df
        unsv_learning_df = df

        
    # saving as csv
    csv_dir = data_dir/data_type/'csvs'
    ensure_folder(csv_dir)
    unsv_learning_df.to_csv(csv_dir/'unsv_learning.csv',index=False)
    metric_learning_df.to_csv(csv_dir/'metric_learning.csv',index=False)

    # csv to refer as example at the time of inference
    if data_type == 'mvtec_ad':
        df_example = df[['Image', 'Category', 'Anomaly', 'OrgSplit']].copy()
    else:
        df_example = df[['Image', 'Category', 'OrgSplit']].copy()
    df_example.to_csv(csv_dir/'example.csv', index=False)
        
    return

if __name__ == '__main__':
    current_file_path = os.path.realpath(__file__)
    data_dir = Path(current_file_path).parent.parent / 'data'

    print('----- Make CSVs for MNIST -----')
    create_csvs(data_dir, 'mnist')
    print('----- Make CSVs for CIFAR10 -----')
    create_csvs(data_dir, 'cifar10')
    print('----- Make CSVs for MVTec_AD -----')
    create_csvs(data_dir, 'mvtec_ad')