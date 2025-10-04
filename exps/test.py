import data_utils
from option import args_parser
from util import *
import torch
import numpy as np
import pandas as pd
import random
from models import *
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torchvision.models as tmodels
from update import LocalUpdate, LocalTest
import copy
from torch import nn
from math import sqrt
import os
import time
from transformers import ViTModel, ViTConfig
import collections
import time
from collections import Counter
from PIL import Image
import concurrent.futures
import pickle

def test_ours(args, local_model, test_loader):
    correct = 0
    model.eval()
    train_iter = iter(test_loader)
    for batch_idx in range(len(train_iter)):
        images, labels = next(train_iter)
        images, labels = images.to(args.device).float(), labels.to(args.device).long()
        rep, logits = model(images)
        pred = logits.data.max(1)[1]
        correct += pred.eq(labels.view(-1)).sum().item()
    acc = correct / len(test_loader.dataset)  
    return acc

if __name__ == "__main__":
    args = args_parser()
    args.dataset = "PACS"
    seeds = [0,1,2]
    test_acc = []
    for seed in seeds:
        args.seed = seed
        accs = []
        if args.dataset == "office":
            args.num_classes = 10
            args.num_users = 4
            args.size = 64
            args.batch = 32
            datasets_name = ["amazon", "caltech", "dslr", "webcam"]
            _, test_loader_list = prepare_data_office_feature_noniid(args=args)
        elif args.dataset == "digit":
            args.num_classes = 10
            args.num_users = 5
            args.size = 28
            args.batch = 64
            datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
            _, test_loader_list = prepare_data_digit_feature_noniid(args=args)
        elif args.dataset == "PACS":
            args.num_classes = 7
            args.num_users = 4
            args.size = 64
            args.batch = 32
            datasets_name = ["art_painting", "cartoon", "photo", "sketch"]
            _, test_loader_list = prepare_data_PACS_feature_noniid(args=args)
        print(args)
        for idx, dataset in enumerate(datasets_name):
            weights_path = f"weights/{args.seed}/{args.dataset}/best_local_model_{dataset}.pth"
            # load best model weight
            model = adcol_model(args.num_classes).to(args.device)
            model.load_state_dict(torch.load(weights_path, weights_only=False, map_location=args.device))
            acc = test_ours(args, model, test_loader_list[idx])
            accs.append(acc)
        test_acc.append(accs)
    test_acc = np.array(test_acc)
    for idx, dataset in enumerate(datasets_name):
        best_acc = sum(test_acc[:, idx]) / len(test_acc[:, idx])
        print(f"accuracy of {dataset} in {args.dataset} is {best_acc}")
        
