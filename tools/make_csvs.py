import numpy as np
import os
import pandas as pd
from pathlib import Path
from utils import *

def make_df(data_dir, data_type):
    def make_df_split(data_dir, data_type, split):
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
                    list_of_dicts.append({'Image':img_path, 'Label':sd.name,
                                          'OrgSplit':split})
        _df = pd.DataFrame(list_of_dicts)

        return _df
    
    df_train = make_df_split(data_dir, data_type, 'train')
    df_test  = make_df_split(data_dir, data_type, 'test')
    df = pd.concat([df_train,df_test], ignore_index=True)
    
    return df

def create_csvs(data_dir, data_type):
    if data_type == 'mvtec_ad':
        target = 'CatAnml'
        
        # metlic learning should be evaluated with cross-validation, 
        # cause original train data has no anomaly instances
        df = make_df(data_dir, data_type)
        df_metric_learning = take_stratified_split(df, target, n_splits=10)
        del df_metric_learning[target]
        
        # take train valid split for unsupervised learning (train/valid data consists of normal data)
        test_idx = df[df['OrgSplit']=='test'].index
        df_unsv_learning = take_stratified_split(df, target, n_splits=1, valid_size=0.3, test_idx=test_idx)
        del df_unsv_learning[target]
        
        # saving as csv
        csv_dir = data_dir/data_type/'csvs'
        ensure_folder(csv_dir)
        df_unsv_learning.to_csv(csv_dir/'unsv_learning.csv',index=False)
        df_metric_learning.to_csv(csv_dir/'metric_learning.csv',index=False)

        # csv to refer as example at the time of inference
        df_example = df[['Image', 'Category', 'Anomaly']].copy()
        df_example.to_csv(csv_dir/'example.csv', index=False)
        
        return
    
    else: # WIP
        return

if __name__ == '__main__':
    current_file_path = os.path.realpath(__file__)
    data_dir = Path(current_file_path).parent.parent / 'data'

    #print('----- Make CSVs for MNIST -----')
    #create_csvs(data_dir, 'mnist')
    #print('----- Make CIFAR10 for MNIST -----')
    #create_csvs(data_dir, 'cifar10')
    print('----- Make CSVs for MVTec_AD -----')
    create_csvs(data_dir, 'mvtec_ad')