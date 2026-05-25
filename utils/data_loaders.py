import torch
import random
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, MNIST
import warnings
import os
import PIL
import numpy as np
import utils.autoaugment as autoaugment
from utils.autoaugment import CIFAR10Policy, ImageNetPolicy
import spikingjelly.datasets
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS

warnings.filterwarnings('ignore')


def build_cifar(cutout=False, use_cifar10=True, download=False):
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), CIFAR10Policy()]
    aug.append(transforms.ToTensor())

    if cutout:
        aug.append(autoaugment.Cutout(n_holes=1, length=16))

    if use_cifar10:
        aug.append(
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = CIFAR10(root='./data/',
                                train=True, download=download, transform=transform_train)
        val_dataset = CIFAR10(root='./data/',
                              train=False, download=download, transform=transform_test)

    else:
        aug.append(
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = CIFAR100(root='/data/dataset/CIFAR100/',
                                 train=True, download=download, transform=transform_train)
        val_dataset = CIFAR100(root='/data/dataset/CIFAR100/',
                               train=False, download=download, transform=transform_test)

    return train_dataset, val_dataset

def build_mnist(download=False):
    train_dataset = MNIST(root='./raw/',
                             train=True, download=download, transform=transforms.ToTensor())
    val_dataset = MNIST(root='./raw/',
                           train=False, download=download, transform=transforms.ToTensor())
    return train_dataset, val_dataset


class DVSCifar10(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.resize = transforms.Resize(size=(48, 48))  # 48 48
        self.tensorx = transforms.ToTensor()
        self.imgx = transforms.ToPILImage()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = torch.load(self.root + '/{}.pt'.format(index))
        # print(data.shape)
        # if self.train:
        new_data = []
        for t in range(data.size(0)):
            new_data.append(self.tensorx(self.resize(self.imgx(data[t,...]))))
        data = torch.stack(new_data, dim=0)
        if self.transform is not None:
            flip = random.random() > 0.5
            if flip:
                data = torch.flip(data, dims=(3,))
            off1 = random.randint(-5, 5)
            off2 = random.randint(-5, 5)
            data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))

        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target.long().squeeze(-1)

    def __len__(self):
        return len(os.listdir(self.root))

def build_dvs128(T):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(128, scale=(0.6, 1.0), interpolation=PIL.Image.NEAREST),
        transforms.Resize(size=(48, 48)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=30),
        spikingjelly.datasets.RandomTemporalDelete(T_remain=T, batch_first=False),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(size=(48, 48)),
    ])
    train_set = DVS128Gesture(root='/data/dataset/DVS128Gesture', train=True, data_type='frame', frames_number=T, #
                              split_by='number')
    test_set = DVS128Gesture(root='/data/dataset/DVS128Gesture', train=False, data_type='frame', frames_number=T,  # /data/dataset/DVS128Gesture
                             split_by='number')

    trainset, testset = packaging_class(train_set, transform_train), packaging_class(test_set, transform_test)
    return trainset, testset

class packaging_class(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.transform = transform
        self.dataset = dataset

    def __getitem__(self, index):
        data, label = self.dataset[index]
        data = torch.FloatTensor(data)
        if self.transform:
            data = self.transform(data)

        return data, label

    def __len__(self):
        return len(self.dataset)
def trans_t(data):
    # print(data.shape)
    # exit(0)
    data = transforms.RandomResizedCrop(128, scale=(0.7, 1.0), interpolation=PIL.Image.NEAREST)(data)
    resize = transforms.Resize(size=(48, 48))  # 48 48
    data = resize(data).float()
    flip = np.random.random() > 0.5
    if flip:
        data = torch.flip(data, dims=(3,))
    data = function_nda(data)
    return data.float()

def trans(data):
    resize = transforms.Resize(size=(48, 48))  # 48 48
    data = resize(data).float()
    return data.float()

def build_dvscifar10(path='/data/dataset/CIFAR10DVS', T=10):    #

    train_path = path + '/train/Train' + str(T)
    test_path = path + '/test/Test' + str(T)

    if os.path.exists(train_path) and os.path.exists(test_path):
        trainset = torch.load(train_path)
        testset = torch.load(test_path)
        print('Load DVSCIFAR10 success')

    else:
        dataset = CIFAR10DVS(root=path, data_type='frame', frames_number=T, split_by='number')
        trainset, testset = spikingjelly.datasets.split_to_train_test_set(train_ratio=0.9, origin_dataset=dataset,
                                                                 num_classes=10)
        trainset, testset = packaging_class(trainset, trans_t), packaging_class(testset, trans)

        torch.save(trainset, train_path)
        torch.save(testset, test_path)

    return trainset, testset

def build_imagenet():
    root = '/data/dataset/ImageNet/'
    train_root = os.path.join(root, 'train')
    val_root = os.path.join(root, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # data augmentation
    crop_scale = 0.08
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    train_dataset = datasets.ImageFolder(
        train_root,
        transform=train_transforms)

    val_dataset = ImageFolder(
        val_root,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize]))
    return train_dataset, val_dataset

from torch.utils.data import Dataset, DataLoader
from torchvision import models, utils, datasets, transforms
import numpy as np
import sys
import os
from PIL import Image


class TinyImageNet_load(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images;

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(self.train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt

def build_tiny_imagenet():
    aug = [transforms.RandomCrop(64, padding=8), transforms.RandomHorizontalFlip(), #ImageNetPolicy(),
           transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    transform_train = transforms.Compose(aug)
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    root = '/data/dataset/tiny-imagenet-200/'
    train_dataset = ImageFolder(os.path.join(root,'train'), transform_train)
    val_dataset = TinyImageNet_load(root, train=False, transform=transform_test)
    # val_dataset = ImageFolder(os.path.join(root, 'val'), transform_test)

    return train_dataset, val_dataset


if __name__ == '__main__':
    train_set, test_set = build_dvs128(T=16)
