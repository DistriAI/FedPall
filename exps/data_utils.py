import sys, os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import pickle
from collections import defaultdict
from collections import Counter

class DigitsDataset(Dataset):
    def __init__(self, data_path, channels, percent=0.1, filename=None, train=True, transform=None):
        if filename is None:
            if train:
                if percent >= 0.1:
                    for part in range(int(percent*10)):
                        if part == 0:
                            self.images, self.labels = np.load(os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
                        else:
                            images, labels = np.load(os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
                            self.images = np.concatenate([self.images,images], axis=0)
                            self.labels = np.concatenate([self.labels,labels], axis=0)
                else:
                    self.images, self.labels = np.load(os.path.join(data_path, 'partitions/train_part0.pkl'), allow_pickle=True)
            else:
                self.images, self.labels = np.load(os.path.join(data_path, 'test.pkl'), allow_pickle=True)
        else:
            self.images, self.labels = np.load(os.path.join(data_path, filename), allow_pickle=True)

        self.transform = transform
        self.channels = channels
        self.labels = self.labels.astype(np.int64).squeeze()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.channels == 1:
            image = Image.fromarray(image, mode='L')
        elif self.channels == 3:
            image = Image.fromarray(image, mode='RGB')
        else:
            raise ValueError("{} channel is not allowed.".format(self.channels))

        if self.transform is not None:
            image = self.transform(image)

        return image, label
    
class DigitsDataset_mul_clients(Dataset):
    def __init__(self, idx,  data_path, channels, percent=0.1, filename=None, train=True, transform=None, noise=False):
        if filename is None:
            if train:
                self.images, self.labels = np.load(os.path.join(data_path, f'partitions/train_part{idx}.pkl'), allow_pickle=True)
                if noise:
                    std_dev = 0.01 * idx
                    noise = np.random.normal(0, std_dev, self.images.shape)
                    self.images = self.images.astype(np.float64) + noise
                data_len = int(self.images.shape[0] * percent*10)
                self.images = self.images[:data_len]
                self.labels = self.labels[:data_len]
            else:
                self.images, self.labels = np.load(os.path.join(data_path, 'test.pkl'), allow_pickle=True)
        else:
            self.images, self.labels = np.load(os.path.join(data_path, filename), allow_pickle=True)

        self.transform = transform
        self.channels = channels
        self.labels = self.labels.astype(np.int64).squeeze()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.channels == 1:
            image = Image.fromarray(image, mode='L')
        elif self.channels == 3:
            image = Image.fromarray(image, mode='RGB')
        else:
            raise ValueError("{} channel is not allowed.".format(self.channels))

        if self.transform is not None:
            image = self.transform(image)

        return image, label
    
class DomainNetDataset(Dataset):
    def __init__(self, base_path, site, transform=None, train=True):
        if train:
            self.paths = [path for path,_ in np.load('../data/DomainNet/{}_train.pkl'.format(site), allow_pickle=True)]
            self.text_labels = [label for _, label in np.load('../data/DomainNet/{}_train.pkl'.format(site), allow_pickle=True)]
        else:
            self.paths = [path for path,_ in np.load('../data/DomainNet/{}_test.pkl'.format(site), allow_pickle=True)]
            self.text_labels = [label for _, label in np.load('../data/DomainNet/{}_test.pkl'.format(site), allow_pickle=True)]
        temp = ["shoe", "table", "tiger", "snowman", "spreadsheet", "whale", "dog", "squirrel", "golf_club", "tree"]
        label_dict = {t:i for i, t in enumerate(temp)}
        self.labels = [label_dict[text] for text in self.text_labels]
        self.transform = transform
        self.base_path = base_path if base_path is not None else '../data'

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path)
        
        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

class OfficeDataset(Dataset):
    def __init__(self, base_path, site, transform=None, train=True):
        if train:
            self.paths = [path for path,_ in np.load('../data/office_caltech_10/{}_train_new.pkl'.format(site), allow_pickle=True)]
            self.text_labels = [label for _, label in np.load('../data/office_caltech_10/{}_train_new.pkl'.format(site), allow_pickle=True)]
        else:
            self.paths = [path for path,_ in np.load('../data/office_caltech_10/{}_test_new.pkl'.format(site), allow_pickle=True)]
            self.text_labels = [label for _, label in np.load('../data/office_caltech_10/{}_test_new.pkl'.format(site), allow_pickle=True)]
        label_dict={'back_pack':0, 'bike':1, 'calculator':2, 'headphones':3, 'keyboard':4, 'laptop_computer':5, 'monitor':6, 'mouse':7, 'mug':8, 'projector':9}
        self.labels = [label_dict[text] for text in self.text_labels]
        self.transform = transform
        self.base_path = base_path if base_path is not None else '../data'

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.normpath(os.path.join(self.base_path, self.paths[idx])).replace("\\", '/')
        label = self.labels[idx]
        image = Image.open(img_path)
        
        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

class PACSDataset(Dataset):
    def __init__(self, base_path, site, transform=None, train=True):
        if train:
            self.paths = [path for path,_ in np.load('../data/PACS/{}_train.pkl'.format(site), allow_pickle=True)]
            self.text_labels = [label for _, label in np.load('../data/PACS/{}_train.pkl'.format(site), allow_pickle=True)]
        else:
            self.paths = [path for path,_ in np.load('../data/PACS/{}_test.pkl'.format(site), allow_pickle=True)]
            self.text_labels = [label for _, label in np.load('../data/PACS/{}_test.pkl'.format(site), allow_pickle=True)]
        label_dict={'dog': 0, 'elephant': 1, 'giraffe': 2, 'guitar': 3, 'horse': 4, 'house': 5, 'person': 6}
        self.labels = [label_dict[text] for text in self.text_labels]
        self.transform = transform
        self.base_path = base_path if base_path is not None else '../data'

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.normpath(os.path.join(self.base_path, self.paths[idx])).replace("\\", '/')
        label = self.labels[idx]
        image = Image.open(img_path)
        
        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label
    
class FeatureDataset(Dataset):
    def __init__(self, features_labels, transform=None, train=True):
        self.features, self.ids= [], []
        for i in range(len(features_labels)):
            self.features.append(features_labels[i][0])
            self.ids.append(features_labels[i][1])
        self.features = torch.concat(self.features, dim=0)
        self.ids = torch.concat(self.ids, dim=0)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        ids = self.ids[idx]
        feature = self.features[idx]
        return feature, ids
    
class FeatureDataset_decode(Dataset):
    def __init__(self, args, features_labels, origin_loaders, transform=None, train=True):
        origin_dataset = []
        for idx in range(args.num_users):
            for t in range(len(origin_loaders[idx].dataset)):
                origin_dataset.append(origin_loaders[idx].dataset[t][0])
        self.features, self.ids, self.origin_data = [], [], []
        for i in range(len(features_labels)):
            self.features.append(features_labels[i][0])
            self.ids.append(features_labels[i][1])
        self.features = torch.concat(self.features, dim=0)
        self.ids = torch.concat(self.ids, dim=0)
        self.origin_data = torch.stack(origin_dataset, dim=0)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        ids = self.ids[idx]
        feature = self.features[idx]
        origin_data = self.origin_data[idx]
        return feature, ids, origin_data
