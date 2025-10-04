import copy
import torch
from torchvision import datasets, transforms
import numpy as np
import data_utils
import seaborn as sns
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import pickle
from torch.optim import Optimizer
import torch.nn.functional as F
from collections import Counter
import functools
# from utils.finch import FINCH

def proto_aggregation(local_protos_list, num_list):
    agg_protos_label = dict()
    total_num_list = functools.reduce(lambda x, y: x + y, num_list)
    for idx in range(len(local_protos_list)):
        local_protos = local_protos_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
            else:
                agg_protos_label[label] = [local_protos[label]]
    averaged_protos = {}
    for label, proto_list in agg_protos_label.items():
        for idx, proto in enumerate(proto_list):
            if label not in averaged_protos:
                averaged_protos[label] = proto * num_list[idx][label] / total_num_list[label]
            else:
                averaged_protos[label] += proto * num_list[idx][label] / total_num_list[label]
    return averaged_protos

def get_mean(args, proto_list, nums):
    weighted_protos = {}
    total_counts = {}
    for idx, proto in enumerate(proto_list):
        for cls, p in proto.items():
            if cls not in weighted_protos:
                weighted_protos[cls] = torch.zeros_like(p) 
                total_counts[cls] = 0
            weighted_protos[cls] += F.normalize(p, dim=0) * nums[idx][cls]
            total_counts[cls] += nums[idx][cls]
    global_protos = {}
    for cls in weighted_protos.keys():
        if total_counts[cls] > 0:
            global_protos[cls] = weighted_protos[cls] / total_counts[cls]
    return global_protos

class PerAvgOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(PerAvgOptimizer, self).__init__(params, defaults)

    def step(self, beta=0):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if(beta != 0):
                    p.data.add_(other=d_p, alpha=-beta)
                else:
                    p.data.add_(other=d_p, alpha=-group['lr'])

def prepare_data_digit_feature_noniid(args):
    # Prepare data
    transform_mnist = transforms.Compose([
            transforms.Resize([args.size,args.size]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    transform_svhn = transforms.Compose([
            transforms.Resize([args.size,args.size]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    transform_usps = transforms.Compose([
            transforms.Resize([args.size,args.size]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    transform_synth = transforms.Compose([
            transforms.Resize([args.size,args.size]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    transform_mnistm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    if args.num_users == 5:
        # MNIST
        mnist_trainset =data_utils.DigitsDataset(data_path="../data/Digits/MNIST", channels=1, percent=args.percent, train=True,  transform=transform_mnist)
        mnist_testset = data_utils.DigitsDataset(data_path="../data/Digits/MNIST", channels=1, percent=args.percent, train=False, transform=transform_mnist)
        # SVHN
        svhn_trainset = data_utils.DigitsDataset(data_path='../data/Digits/SVHN', channels=3, percent=args.percent,  train=True,  transform=transform_svhn)
        svhn_testset = data_utils.DigitsDataset(data_path='../data/Digits/SVHN', channels=3, percent=args.percent,  train=False, transform=transform_svhn)
        # USPS
        usps_trainset = data_utils.DigitsDataset(data_path='../data/Digits/USPS', channels=1, percent=args.percent,  train=True,  transform=transform_usps)
        usps_testset = data_utils.DigitsDataset(data_path='../data/Digits/USPS', channels=1, percent=args.percent,  train=False, transform=transform_usps)
        # Synth Digits
        synth_trainset = data_utils.DigitsDataset(data_path='../data/Digits/SynthDigits/', channels=3, percent=args.percent,  train=True,  transform=transform_synth)
        synth_testset = data_utils.DigitsDataset(data_path='../data/Digits/SynthDigits/', channels=3, percent=args.percent,  train=False, transform=transform_synth)
        # MNIST-M
        mnistm_trainset = data_utils.DigitsDataset(data_path='../data/Digits/MNIST_M/', channels=3, percent=args.percent,  train=True,  transform=transform_mnistm)
        mnistm_testset = data_utils.DigitsDataset(data_path='../data/Digits/MNIST_M/', channels=3, percent=args.percent,  train=False, transform=transform_mnistm)
        
        mnist_train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=args.batch, shuffle=True, num_workers=args.number_workers, pin_memory=True)
        mnist_test_loader  = torch.utils.data.DataLoader(mnist_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)
        svhn_train_loader = torch.utils.data.DataLoader(svhn_trainset, batch_size=args.batch,  shuffle=True, num_workers=args.number_workers, pin_memory=True)
        svhn_test_loader = torch.utils.data.DataLoader(svhn_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)
        usps_train_loader = torch.utils.data.DataLoader(usps_trainset, batch_size=args.batch,  shuffle=True, num_workers=args.number_workers, pin_memory=True)
        usps_test_loader = torch.utils.data.DataLoader(usps_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)
        synth_train_loader = torch.utils.data.DataLoader(synth_trainset, batch_size=args.batch,  shuffle=True, num_workers=args.number_workers, pin_memory=True)
        synth_test_loader = torch.utils.data.DataLoader(synth_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)
        mnistm_train_loader = torch.utils.data.DataLoader(mnistm_trainset, batch_size=args.batch,  shuffle=True, num_workers=args.number_workers, pin_memory=True)
        mnistm_test_loader = torch.utils.data.DataLoader(mnistm_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)

        train_loaders = [mnist_train_loader, svhn_train_loader, usps_train_loader, synth_train_loader, mnistm_train_loader]
        test_loaders  = [mnist_test_loader, svhn_test_loader, usps_test_loader, synth_test_loader, mnistm_test_loader]
    else:
        train_loaders, test_loaders = [], []
        for idx in range(args.num_users // 5):
            # MNIST
            mnist_trainset = data_utils.DigitsDataset_mul_clients(idx, data_path="../data/Digits/MNIST", channels=1, percent=args.percent, train=True,  transform=transform_mnist, noise=args.noise)
            # SVHN
            svhn_trainset = data_utils.DigitsDataset_mul_clients(idx, data_path='../data/Digits/SVHN', channels=3, percent=args.percent,  train=True,  transform=transform_svhn, noise=args.noise)
            # USPS
            usps_trainset = data_utils.DigitsDataset_mul_clients(idx, data_path='../data/Digits/USPS', channels=1, percent=args.percent,  train=True,  transform=transform_usps, noise=args.noise)
            # Synth Digits
            synth_trainset = data_utils.DigitsDataset_mul_clients(idx, data_path='../data/Digits/SynthDigits/', channels=3, percent=args.percent,  train=True,  transform=transform_synth, noise=args.noise)
            # MNIST-M
            mnistm_trainset = data_utils.DigitsDataset_mul_clients(idx, data_path='../data/Digits/MNIST_M/', channels=3, percent=args.percent,  train=True,  transform=transform_mnistm, noise=args.noise)
            mnist_train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=args.batch, shuffle=True)
            svhn_train_loader = torch.utils.data.DataLoader(svhn_trainset, batch_size=args.batch,  shuffle=True)
            usps_train_loader = torch.utils.data.DataLoader(usps_trainset, batch_size=args.batch,  shuffle=True)
            synth_train_loader = torch.utils.data.DataLoader(synth_trainset, batch_size=args.batch,  shuffle=True)
            mnistm_train_loader = torch.utils.data.DataLoader(mnistm_trainset, batch_size=args.batch,  shuffle=True)
            train_loaders.extend([mnist_train_loader, svhn_train_loader, usps_train_loader, synth_train_loader, mnistm_train_loader])
        
        mnist_testset = data_utils.DigitsDataset_mul_clients(idx, data_path="../data/Digits/MNIST", channels=1, percent=args.percent, train=False, transform=transform_mnist, noise=args.noise)
        svhn_testset = data_utils.DigitsDataset_mul_clients(idx, data_path='../data/Digits/SVHN', channels=3, percent=args.percent,  train=False, transform=transform_svhn, noise=args.noise)
        usps_testset = data_utils.DigitsDataset_mul_clients(idx, data_path='../data/Digits/USPS', channels=1, percent=args.percent,  train=False, transform=transform_usps, noise=args.noise)
        synth_testset = data_utils.DigitsDataset_mul_clients(idx, data_path='../data/Digits/SynthDigits/', channels=3, percent=args.percent,  train=False, transform=transform_synth, noise=args.noise)
        mnistm_testset = data_utils.DigitsDataset_mul_clients(idx, data_path='../data/Digits/MNIST_M/', channels=3, percent=args.percent,  train=False, transform=transform_mnistm, noise=args.noise)
        mnist_test_loader  = torch.utils.data.DataLoader(mnist_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)
        svhn_test_loader = torch.utils.data.DataLoader(svhn_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)
        usps_test_loader = torch.utils.data.DataLoader(usps_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)
        synth_test_loader = torch.utils.data.DataLoader(synth_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)
        mnistm_test_loader = torch.utils.data.DataLoader(mnistm_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)
        
        test_loaders.extend([mnist_test_loader, svhn_test_loader, usps_test_loader, synth_test_loader, mnistm_test_loader])
    return train_loaders, test_loaders

def prepare_data_office_feature_noniid(args):
    data_base_path = '../data/office_caltech_10'
    transform_office = transforms.Compose([
            transforms.Resize([args.size, args.size]),            
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
            transforms.Resize([args.size, args.size]),            
            transforms.ToTensor(),
    ])
    
    # amazon
    amazon_trainset = data_utils.OfficeDataset(data_base_path, 'amazon', transform=transform_office)
    amazon_testset = data_utils.OfficeDataset(data_base_path, 'amazon', transform=transform_test, train=False)
    # caltech
    caltech_trainset = data_utils.OfficeDataset(data_base_path, 'caltech', transform=transform_office)
    caltech_testset = data_utils.OfficeDataset(data_base_path, 'caltech', transform=transform_test, train=False)
    # dslr
    dslr_trainset = data_utils.OfficeDataset(data_base_path, 'dslr', transform=transform_office)
    dslr_testset = data_utils.OfficeDataset(data_base_path, 'dslr', transform=transform_test, train=False)
    # webcam
    webcam_trainset = data_utils.OfficeDataset(data_base_path, 'webcam', transform=transform_office)
    webcam_testset = data_utils.OfficeDataset(data_base_path, 'webcam', transform=transform_test, train=False)

    amazon_train_loader = torch.utils.data.DataLoader(amazon_trainset, batch_size=args.batch, shuffle=True, num_workers=args.number_workers, pin_memory=True)
    amazon_test_loader = torch.utils.data.DataLoader(amazon_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)

    caltech_train_loader = torch.utils.data.DataLoader(caltech_trainset, batch_size=args.batch, shuffle=True, num_workers=args.number_workers, pin_memory=True)
    caltech_test_loader = torch.utils.data.DataLoader(caltech_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)

    dslr_train_loader = torch.utils.data.DataLoader(dslr_trainset, batch_size=args.batch, shuffle=True, num_workers=args.number_workers, pin_memory=True)
    dslr_test_loader = torch.utils.data.DataLoader(dslr_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)

    webcam_train_loader = torch.utils.data.DataLoader(webcam_trainset, batch_size=args.batch, shuffle=True, num_workers=args.number_workers, pin_memory=True)
    webcam_test_loader = torch.utils.data.DataLoader(webcam_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)
    
    train_loaders = [amazon_train_loader, caltech_train_loader, dslr_train_loader, webcam_train_loader]
    test_loaders = [amazon_test_loader, caltech_test_loader, dslr_test_loader, webcam_test_loader]
    return train_loaders, test_loaders

def prepare_data_PACS_feature_noniid(args):
    data_base_path = '../data/PACS'
    transform_PACS= transforms.Compose([
            transforms.Resize([args.size, args.size]),            
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
            transforms.Resize([args.size, args.size]),            
            transforms.ToTensor(),
    ])
    dataset_name = ["art_painting", "cartoon", "photo", "sketch"]
    train_loaders, test_loaders = [], []
    for dataset in dataset_name:
        train_set = data_utils.PACSDataset(data_base_path, dataset, transform=transform_PACS)
        test_set = data_utils.PACSDataset(data_base_path, dataset, transform=transform_test, train=False)
        train_loaders.append(torch.utils.data.DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.number_workers, pin_memory=True))
        test_loaders.append(torch.utils.data.DataLoader(test_set, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True))
    return train_loaders, test_loaders

def average_protos(protos):
    """
    Average the protos for each local user
    """
    agg_protos = {}
    for [label, proto_list] in protos.items():
        proto = np.stack(proto_list)
        agg_protos[label] = np.mean(proto, axis=0)

    return agg_protos

# def cluster_protos_finch(protos_label_dict):
#     agg_protos = {}
#     num_p = 0
#     for [label, proto_list] in protos_label_dict.items():
#         proto_list = np.stack(proto_list)
#         c, num_clust, req_c = FINCH(proto_list, initial_rank=None, req_clust=None, distance='cosine',
#                                     ensure_early_exit=False, verbose=False)
#         num_protos, num_partition = c.shape
#         class_cluster_list = []
#         for idx in range(num_protos):
#             class_cluster_list.append(c[idx, -1])
#         class_cluster_array = np.array(class_cluster_list)
#         uniqure_cluster = np.unique(class_cluster_array).tolist()
#         agg_selected_proto = []

#         for _, cluster_index in enumerate(uniqure_cluster):
#             selected_array = np.where(class_cluster_array == cluster_index)
#             selected_proto_list = proto_list[selected_array]
#             cluster_proto_center = np.mean(selected_proto_list, axis=0)
#             agg_selected_proto.append(cluster_proto_center)

#         agg_protos[label] = agg_selected_proto
#         num_p += num_clust[-1]
#     return agg_protos, num_p / len(protos_label_dict)

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w)
    for key in w[0].keys():
        for i in range(1, len(w)):
            w_avg[0][key] += w[i][key]
        w_avg[0][key] = torch.true_divide(w_avg[0][key], len(w))
        for i in range(1, len(w)):
            w_avg[i][key] = w_avg[0][key]
    return w_avg

def local_cluster_collect(local_cluster_protos):
    global_collected_protos = {}
    for [idx, cluster_protos_label] in local_cluster_protos.items():
        for [label, cluster_protos_list] in cluster_protos_label.items():
            for i in range(len(cluster_protos_list)):
                if label in global_collected_protos.keys():
                    global_collected_protos[label].append(cluster_protos_list[i])
                else:
                    global_collected_protos[label] = [cluster_protos_list[i]]
    return global_collected_protos

def proto_aggregation_cluster(global_protos_list):
    agg_protos_label = dict()
    for label in global_protos_list.keys():
        for i in range(len(global_protos_list[label])):
            if label in agg_protos_label:
                agg_protos_label[label].append(global_protos_list[label][i])
            else:
                agg_protos_label[label] = [global_protos_list[label][i]]

    for [label, proto_list] in agg_protos_label.items():
        # print(len(proto_list))
        if len(proto_list) > 1:
            proto = 0 * proto_list[0]
            for i in proto_list:
                proto += i
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0]

    return agg_protos_label