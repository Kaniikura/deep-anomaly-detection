from collections import defaultdict
import numpy as np
import os
from pathlib import Path
import PIL
from torchvision import datasets, transforms

from utils import *

    
# ------ cited and partially modified from https://github.com/daisukelab/dl-cliche/tree/master/dlcliche    
def prepare_MNIST(data_path=Path('../data/mnist')):
    """
    Download and restructure dataset as images under:
        data_path/images/('train' or 'valid')/(class)
    Where filenames are:
        img(class)_(count index).png
    
    Returns:
        Created data path.
    """
    def have_already_been_done():
        return (data_path/'images').is_dir()
    def build_images_folder(data_root, X, labels, dest_folder):
        images = data_path/'images'
        for i, (x, y) in tqdm(enumerate(zip(X, labels))):
            folder = images/dest_folder/f'{y}'
            ensure_folder(folder)
            x = x.numpy()
            image = np.stack([x for ch in range(3)], axis=-1)
            PIL.Image.fromarray(image).save(folder/f'img{y}_{i:06d}.png')

    train_ds = datasets.MNIST(data_path, train=True, download=True,
                          transform=transforms.Compose([
                              transforms.Normalize((0.1307,), (0.3081,))
                          ]))
    valid_ds = datasets.MNIST(data_path, train=False,
                              transform=transforms.Compose([
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ]))

    if not have_already_been_done():
        build_images_folder(data_root=data_path, X=train_ds.train_data,
                            labels=train_ds.train_labels, dest_folder='train')
        build_images_folder(data_root=data_path, X=valid_ds.test_data,
                            labels=valid_ds.test_labels, dest_folder='test')

    return data_path/'images'

def prepare_CIFAR10(data_path=Path('../data/cifar10')):
    """
    Download and restructure CIFAR10 dataset as images under:
        data_path/images/('train' or 'valid')/(class)
    Where filenames are:
        img(class)_(count index).png
    Returns:
        Restructured data path.
    """
    def have_already_been_done():
        return (data_path/'images').is_dir()
    def build_images_folder(data_root, X, labels, dest_folder):
        images = data_path/'images'
        for i, (x, y) in tqdm(enumerate(zip(X, labels))):
            folder = images/dest_folder/f'{classes[y]}'
            ensure_folder(folder)
            PIL.Image.fromarray(x).save(folder/f'img{y}_{i:06d}.png')

    train_ds = datasets.CIFAR10(data_path, train=True, download=True,
                          transform=transforms.Compose([
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                          ]))
    valid_ds = datasets.CIFAR10(data_path, train=False,
                          transform=transforms.Compose([
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                          ]))

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    if not have_already_been_done():
        build_images_folder(data_root=data_path, X=train_ds.data,
                            labels=train_ds.targets, dest_folder='train')
        build_images_folder(data_root=data_path, X=valid_ds.data,
                            labels=valid_ds.targets, dest_folder='test')

    return data_path/'images'

def prepare_MVTecAD(data_path=Path('../data/mvtec_ad'), exclude_toothbrush=True, chmod=True):
    """
    Download and extract MVTec Anomaly Detection (MVTec AD) dataset.
    Files will be placed as follows:
        data_path/original ... all extracted original folders/files, and tar archive.
    Arguments:
        data_path: Path to place data.
        exclude_toothbrush: True if excluding toothbrush from return value `testcases`. It has only one test class.
    Returns:
        data_path: Input data_path as is.
        testcases: List of test cases in the dataset.
    """
    def have_already_been_done():
        return org_path.is_dir()

    def build_train_test_MVTecAD(path):
        """
        Builds train & test folders to adjust to MNIST and CIFAR10.
        """
        
        subs = {}
        cats = [ p.name for p in (path/'original').glob('[A-Za-z0-9]*') if p.is_dir()]
        for c in cats:
            cur = sorted([d.name for d in (path/f'original/{c}/test').glob('[A-Za-z0-9]*')
                        if d.name != 'good'])
            subs.update({c: cur})
        # Build train & test folders
        new_path = path/'images'
        for cat in cats:
            ensure_delete(new_path/'train'/cat/'good')
            ensure_delete(new_path/'test'/cat/'good')
            # Copy samples
            copy_any(path/f'original/{cat}/train/good', new_path/'train'/cat/'good')
            copy_any(path/f'original/{cat}/test/good',  new_path/'test'/cat/'good')
            for sub in subs[cat]:
                ensure_delete(new_path/'test'/cat/sub)
                # Copy samples
                copy_any(path/f'original/{cat}/test/{sub}', new_path/'test'/cat/sub)
           

    url = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
    md5 = 'eefca59f2cede9c3fc5b6befbfec275e'
    org_path = data_path/'original'
    file_path = org_path/url.split('/')[-1]

    if not have_already_been_done():
        if not (file_path).is_file():
            datasets.utils.download_url(url, str(org_path), file_path.name, md5)
        ensure_folder(org_path)
        print(f'Extracting {file_path}')
        extract_archive(str(file_path), str(org_path), remove_finished=False)

        if chmod:
            chmod_tree_all(org_path, mode=0o775)

    build_train_test_MVTecAD(data_path)

    testcases = sorted([d.name for d in org_path.iterdir() if d.is_dir()])
    if exclude_toothbrush:
        testcases = [tc for tc in testcases if tc != 'toothbrush']

    return data_path, testcases
# ------ end of citation from https://github.com/daisukelab/dl-cliche/tree/master/dlcliche

if __name__ == '__main__':
    current_file_path = os.path.realpath(__file__)
    posix_path = Path(current_file_path).parent.parent / 'data'

    print('----- Create MNIST directory -----')
    prepare_MNIST(posix_path / 'mnist')
    print('----- Create CIFAR10 directory -----')
    prepare_CIFAR10(posix_path / 'cifar10')
    print('----- Create MVTec_AD directory -----')
    prepare_MVTecAD(posix_path / 'mvtec_ad')