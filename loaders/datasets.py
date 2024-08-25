import os
import torch
import numpy as np
from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms
import glob
import cv2 as cv

imagenet_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def resize(label):
    label = label / 255  # 0-1
    label = label.reshape([1, label.shape[0], label.shape[1]])
    label = np.concatenate((1 - label, label), axis=0)  # 类别数为2   [2, H, W]
    return label

def load(split_path):
    res = []
    with open(split_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            v = line.replace("\n", "")
            v = v.replace("\\", "/")
            res.append(v)
    return res

class GZDataset(data.Dataset):
    def __init__(self, split, supervised_train=True, transforms_unsup=None):
        assert split in ["train", "test"]
        self.split = split
        self.supervised_train = supervised_train
        self.transforms = transforms_unsup
        root_dir = "/data/sqd/GZ_CD256"
        # root_dir = r"D:\Project\change_detection\data\GZ_CD256"
        root_dir = os.path.join(root_dir, split)
        # image
        T1_image_path = glob.glob(root_dir + '/A' + '/*.tif')
        T2_image_path = glob.glob(root_dir + '/B' + '/*.tif')
        label_path = glob.glob(root_dir + '/label' + '/*.png')

        T1_image_path.sort()
        T2_image_path.sort()
        label_path.sort()

        self.T1_image_path = T1_image_path
        self.T2_image_path = T2_image_path
        self.label_path = label_path


    def __getitem__(self, idx):
        sample = {}
        image1 = Image.open(self.T1_image_path[idx])
        image2 = Image.open(self.T2_image_path[idx])
        label = cv.imread(self.label_path[idx], 0)
        label = (label != 0).astype('uint8')
        label = torch.from_numpy(label).long()

        if self.supervised_train or self.split == "test":
            image1 = imagenet_preprocess(image1)
            image2 = imagenet_preprocess(image2)
            sample['image'] = [image1, image2]
            sample['labels'] = label
        else:                                                # 'sample0' no augmentation  'sample1' strong augmentation
            sample0 = {}
            sample1 = {}
            image11 = imagenet_preprocess(image1)
            image22 = imagenet_preprocess(image2)
            sample0['image'] = [image11, image22]
            image1 = self.transforms(image1)
            image2 = self.transforms(image2)
            sample1['image'] = [image1, image2]

            sample['sample0'] = sample0
            sample['sample1'] = sample1

        return sample

    def __len__(self):
        return len(self.T1_image_path)


class S2lookingDataset_mini(data.Dataset):
    def __init__(self, split, supervised_train=True, transforms_unsup=None, change_type="T1"):
        assert split in ["train", "test"]
        self.split = split
        self.supervised_train = supervised_train
        self.transforms = transforms_unsup
        root_dir = "/data0/qidi/S2looking"

        txt_dir = os.path.join(root_dir, split+"_"+ change_type +".txt")  # train256_T1.txt  test.txt

        self.ids = load(txt_dir)
        self.T1_image_path = []
        self.T2_image_path = []
        self.label_path = []

        for i in self.ids:
            self.T1_image_path.append(os.path.join(root_dir, split +"/A", i))
            self.T2_image_path.append(os.path.join(root_dir, split + "/B", i))
            self.label_path.append(os.path.join(root_dir, split + "/label", i))

    def __getitem__(self, idx):
        sample = {}
        image1 = Image.open(self.T1_image_path[idx])
        image2 = Image.open(self.T2_image_path[idx])
        label = cv.imread(self.label_path[idx], 0)
        label = (label != 0).astype('uint8')
        label = torch.from_numpy(label).long()

        if self.supervised_train or self.split == "test":
            image1 = imagenet_preprocess(image1)
            image2 = imagenet_preprocess(image2)
            sample['image'] = [image1, image2]
            sample['labels'] = label
        else:                                                # 'sample0' no augmentation  'sample1' strong augmentation
            sample0 = {}
            sample1 = {}
            image11 = imagenet_preprocess(image1)
            image22 = imagenet_preprocess(image2)
            sample0['image'] = [image11, image22]
            image1 = self.transforms(image1)
            image2 = self.transforms(image2)
            sample1['image'] = [image1, image2]

            sample['sample0'] = sample0
            sample['sample1'] = sample1

        # return sample, self.ids[idx]
        return sample

    def __len__(self):
        return len(self.T1_image_path)


class S2lookingDataset_all(data.Dataset):
    def __init__(self, split, supervised_train=True, transforms_unsup=None):
        assert split in ["train", "test", "val"]
        self.split = split
        self.supervised_train = supervised_train
        self.transforms = transforms_unsup
        root_dir = "/data0/qidi/S2looking"

        root_dir = os.path.join(root_dir, split)
        # image
        T1_image_path = glob.glob(root_dir + '/A' + '/*.png')
        T2_image_path = glob.glob(root_dir + '/B' + '/*.png')
        label_path = glob.glob(root_dir + '/label' + '/*.png')

        T1_image_path.sort()
        T2_image_path.sort()
        label_path.sort()

        # self.ids = load(txt_dir)
        self.T1_image_path = T1_image_path
        self.T2_image_path = T2_image_path
        self.label_path = label_path


    def __getitem__(self, idx):
        sample = {}
        image1 = Image.open(self.T1_image_path[idx])
        image2 = Image.open(self.T2_image_path[idx])
        label = cv.imread(self.label_path[idx], 0)
        label = (label != 0).astype('uint8')
        label = torch.from_numpy(label).long()

        if self.supervised_train or self.split == "test":
            image1 = imagenet_preprocess(image1)
            image2 = imagenet_preprocess(image2)
            sample['image'] = [image1, image2]
            sample['labels'] = label
        else:                                                # 'sample0' no augmentation  'sample1' strong augmentation
            sample0 = {}
            sample1 = {}
            image11 = imagenet_preprocess(image1)
            image22 = imagenet_preprocess(image2)
            sample0['image'] = [image11, image22]
            image1 = self.transforms(image1)
            image2 = self.transforms(image2)
            sample1['image'] = [image1, image2]

            sample['sample0'] = sample0
            sample['sample1'] = sample1

        # return sample, self.ids[idx]
        return sample

    def __len__(self):
        return len(self.T1_image_path)

class LEVIRDataset(data.Dataset):
    def __init__(self, split, supervised_train=True, transforms_unsup=None):
        assert split in ["train", "test", "val"]
        self.split = split
        self.supervised_train = supervised_train
        self.transforms = transforms_unsup
        root_dir = "/data0/qidi/LEVIR-CD256"
        # root_dir = "/data/sqd/LEVIR-CD256"
        # root_dir = r"D:\Project\change_detection\data\GZ_CD256"
        root_dir = os.path.join(root_dir, split)
        # image
        T1_image_path = glob.glob(root_dir + '/A' + '/*.png')
        T2_image_path = glob.glob(root_dir + '/B' + '/*.png')
        label_path = glob.glob(root_dir + '/label' + '/*.png')

        T1_image_path.sort()
        T2_image_path.sort()
        label_path.sort()

        self.T1_image_path = T1_image_path
        self.T2_image_path = T2_image_path
        self.label_path = label_path


    def __getitem__(self, idx):
        sample = {}
        image1 = Image.open(self.T1_image_path[idx])
        image2 = Image.open(self.T2_image_path[idx])
        label = cv.imread(self.label_path[idx], 0)
        label = (label != 0).astype('uint8')
        label = torch.from_numpy(label).long()

        if self.supervised_train or self.split == "test":
            image1 = imagenet_preprocess(image1)
            image2 = imagenet_preprocess(image2)
            sample['image'] = [image1, image2]
            sample['labels'] = label
        else:                                                # 'sample0' no augmentation  'sample1' strong augmentation
            sample0 = {}
            sample1 = {}
            image11 = imagenet_preprocess(image1)
            image22 = imagenet_preprocess(image2)
            sample0['image'] = [image11, image22]
            image1 = self.transforms(image1)
            image2 = self.transforms(image2)
            sample1['image'] = [image1, image2]

            sample['sample0'] = sample0
            sample['sample1'] = sample1

        return sample

    def __len__(self):
        return len(self.T1_image_path)