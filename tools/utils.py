from collections import defaultdict
import gzip
import numpy as np
import os
import pandas as pd
from pathlib import Path
import PIL
import shutil
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import tarfile
from torchvision import datasets, transforms
from tqdm import tqdm
import zipfile

    
# ------ borrowing from https://github.com/pytorch/vision/blob/master/torchvision/datasets/utils.py
def _is_tar(filename):
    return filename.endswith(".tar")


def _is_targz(filename):
    return filename.endswith(".tar.gz")


def _is_tarxz(filename):
    return filename.endswith(".tar.xz")


def _is_gzip(filename):
    return filename.endswith(".gz") and not filename.endswith(".tar.gz")


def _is_zip(filename):
    return filename.endswith(".zip")

    
def extract_archive(from_path, to_path=None, remove_finished=False):
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if _is_tar(from_path):
        with tarfile.open(from_path, 'r') as tar:
            tar.extractall(path=to_path)
    elif _is_targz(from_path):
        with tarfile.open(from_path, 'r:gz') as tar:
            tar.extractall(path=to_path)
    elif _is_tarxz(from_path):
        with tarfile.open(from_path, 'r:xz') as tar:
            tar.extractall(path=to_path)
    elif _is_gzip(from_path):
        to_path = os.path.join(to_path, os.path.splitext(os.path.basename(from_path))[0])
        with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif _is_zip(from_path):
        with zipfile.ZipFile(from_path, 'r') as z:
            z.extractall(to_path)
    else:
        raise ValueError("Extraction of {} not supported".format(from_path))

    if remove_finished:
        os.remove(from_path)
# ------ end of borrowing from https://github.com/pytorch/vision/blob/master/torchvision/datasets/utils.py

# ------ cited and partially modified from https://github.com/daisukelab/dl-cliche/tree/master/dlcliche
def copy_file(src, dst):
    """Copy source file to destination file."""
    assert Path(src).is_file()
    shutil.copy(str(src), str(dst))

def _copy_any(src, dst, symlinks):
        if Path(src).is_dir():
            assert not Path(dst).exists()
            shutil.copytree(src, dst, symlinks=symlinks)
        else:
            copy_file(src, dst)


def copy_any(src, dst, symlinks=True):
    """Copy any file or folder recursively.
    Source file can be list/array of files.
    """
    do_list_item(_copy_any, src, dst, symlinks)



def do_list_item(func, src, *prms):
    if isinstance(src, (list, tuple, np.ndarray)):
        result = True
        for element in src:
            result = do_list_item(func, element, *prms) and result
        return result
    else:
        return func(src, *prms)

def ensure_folder(folder):
    """Make sure a folder exists."""
    Path(folder).mkdir(exist_ok=True, parents=True)

def ensure_delete(folder_or_file):
        anything = Path(folder_or_file)
        if anything.is_dir():
            shutil.rmtree(str(folder_or_file))
        elif anything.exists():
            anything.unlink()
    
def chmod_tree_all(tree_root, mode=0o775):
    """Change permission for all the files or directories under the tree_root."""
    for root, dirs, files in os.walk(tree_root):
        for d in dirs:
            os.chmod(os.path.join(root, d), mode)
        for f in files:
            os.chmod(os.path.join(root, f), mode)

# ------ end of borrowing from https://github.com/daisukelab/dl-cliche/tree/master/dlcliche

def take_stratified_split(_df, target, n_splits=10, valid_size=None, test_idx=None):
    df = _df.copy()
    if test_idx is not None:
        df_test = df.iloc[test_idx].copy()
        df = df[~df.index.isin(test_idx)]
        
    y = df[target]
    seed = 1234
    
    if n_splits==1: # take train valid split
        assert valid_size is not None
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=valid_size, random_state=seed)
        sss.get_n_splits(df, y)
        train_index, valid_index  = sss.split(df, y).__next__()
        df.loc[train_index, 'Fold'] = 0 # fold 0 is train data
        df.loc[valid_index, 'Fold'] = 1 # fold 1 is validation data
        if test_idx is not None:
            df_test['Fold'] = 2 # fold 2 is test data
            df = pd.concat([df,df_test], ignore_index=True)
        
    else :
        assert valid_size is None
        skf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
        for i, (_, valid_index) in enumerate(skf.split(df, y)):
            df.loc[valid_index, 'Fold'] = i
        if test_idx is not None:
            df_test['Fold'] = i+1 # a fold which has max value is corresponding to test data
            df = pd.concat([df,df_test], ignore_index=True)
            
    df.Fold = df.Fold.astype(int)
    
    return df