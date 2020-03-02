from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import albumentations as albu

import dlcommon

def get_training_augmentation(resize_to=(256,256), crop_size=(224, 224), do_flip = False):
    print('[get_training_augmentation] resize_to:', resize_to) 
    #print('[get_training_augmentation] crop_size:', crop_size) 

    train_transform = [
        #albu.RandomScale(scale_limit=(0, 0.1),p=0.75), 
        #albu.Rotate(limit=10, p=0.75),
        #albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=(-0.2, 0.25), p=0.75),
        albu.Resize(*resize_to),
        albu.Normalize(),
    ]

    return albu.Compose(train_transform)

def get_test_augmentation(resize_to=(256,256)):
    test_transform = [
        albu.Resize(*resize_to),
        albu.Normalize(),
    ]
    return albu.Compose(test_transform)

@dlcommon.TRANSFORMS.register
def metric_transform(split, resize_to=(256,256), do_flip=False, tta=1, **_):
    if isinstance(resize_to, str):
        resize_to = eval(resize_to)
    
    print('[metric__transform] resize_to:', resize_to)
    print('[metric__transform] do_flip:', do_flip)
    print('[metric_transform] tta:', tta)

    train_aug = get_training_augmentation(resize_to, do_flip)
    test_aug = get_test_augmentation(resize_to)

    def transform(image):
        if split == 'train':
            augmented = train_aug(image=image)
        else:
            augmented = test_aug(image=image)

        if tta > 1 and (split!='get_embeddings'):
            images = []
            images.append(augmented['image'])
            images.append(test_aug(image=np.fliplr(image))['image'])
            if tta > 2:
                images.append(test_aug(image=np.flipud(image))['image'])
            if tta > 3:
                images.append(test_aug(image=np.flipud(np.fliplr(image)))['image'])
            image = np.stack(images, axis=0)
            image = np.transpose(image, (0,3,1,2))
        else:
            image = augmented['image']
            image = np.transpose(image, (2,0,1))
           
        return image

    return transform
