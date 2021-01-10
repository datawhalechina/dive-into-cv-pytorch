import os
import cv2
import random
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms


def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(256, 256)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


class Building_Dataset(Dataset):  # 继承Dataset类

    def __init__(self, dataset_info):

        self.dataset_info = dataset_info        
    
        self.mean_list = [0.485, 0.456, 0.406]
        self.std_list = [0.229, 0.224, 0.225] 
        self.trans_Normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean_list, std=self.std_list),
        ])  


    def __getitem__(self, index):
        img_path, mask_rle_str = self.dataset_info[index]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #imgs = imgs.astype(np.float32)
        img = self.trans_Normalize(img)

        mask = rle_decode(mask_rle_str, (512, 512))
        masks = np.expand_dims(mask, axis=0)
        masks = torch.FloatTensor(masks)

        return img, masks


    def __len__(self):
        return len(self.dataset_info)


def load_dataset_info(img_dir, anno_path):
    dataset_info = []
    with open(anno_path, 'r') as reader:
        for line in reader:
            items = line.rstrip().split('\t')
            img_name = items[0]
            img_path = os.path.join(img_dir, img_name)
            if len(items) == 2:
                mask_rle_str = items[1]
            else:
                mask_rle_str = ''
            cur_info = [img_path, mask_rle_str]
            dataset_info.append(cur_info)
    return dataset_info


def split_dataset_info(dataset_info, val_percent):
    length = len(dataset_info)
    n = int(length * val_percent)
    random.shuffle(dataset_info)
    train_set = dataset_info[:-n]
    valid_set = dataset_info[-n:]
    return train_set, valid_set

