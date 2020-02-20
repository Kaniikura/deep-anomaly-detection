from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from torchvision import transforms
import dlcommon

def get_training_augmentation(resize_to=(256,256), normalize=True):
    print('[get_training_augmentation] resize_to:', resize_to) 

    train_transform = transforms.Compose([
        lambda x: x.astype(np.uint8),
        transforms.ToPILImage(),
        transforms.Resize(resize_to),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]) if normalize else lambda x:x,
    ])

    return train_transform

def get_test_augmentation(resize_to=(256,256), normalize=True):
    test_transform = transforms.Compose([
        lambda x: x.astype(np.uint8),
        transforms.ToPILImage(),
        transforms.Resize(resize_to),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]) if normalize else lambda x:x,
    ])
    return test_transform

@dlcommon.TRANSFORMS.register
def unsv_transform(split, resize_to=(256,256), normalize=True, tta=1, **_):
    if isinstance(resize_to, str):
        resize_to = eval(resize_to)
    
    print('[unsv_transform] resize_to:', resize_to)
    print('[unsv_transform] normalize:', normalize)
    print('[unsv_transform] tta:', tta)

    train_aug = get_training_augmentation(resize_to,normalize)
    test_aug = get_test_augmentation(resize_to,normalize)

    def transform(image):
        if split == 'train':
            image = train_aug(image)
        else:
            image = test_aug(image)

        if tta > 1:
            images = []
            images.append(image)
            images.append(test_aug(np.fliplr(image)))
            if tta > 2:
                images.append(test_aug(np.flipud(image)))
            if tta > 3:
                images.append(test_aug(np.flipud(np.fliplr(image))))
            image = np.stack(images, axis=0)

        return image

    return transform
