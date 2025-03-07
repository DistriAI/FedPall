import data_utils
from option import args_parser
from utils import *
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

def SingleSet(args, train_loader_list, test_loader_list):
    loader_size = [len(train_loader.dataset) for train_loader in train_loader_list]
    client_weights = [item / sum(loader_size) for item in loader_size]
    if args.dataset == "digit":
        global_model = adcol_model().to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
    elif args.dataset == "office":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["amazon", "caltech", "dslr", "webcam"]  
    elif args.dataset == "PACS":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["art_painting", "cartoon", "photo", "sketch"]
    train_loss = {item: [] for item in datasets_name}
    accuracy_list = {item: [] for item in datasets_name}
    for round in tqdm(range(args.iters)):
        print(f'\n | Global Training Round : {round} |\n')
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                local_node = LocalUpdate(args=args)
                local_models[idx1 * len(datasets_name) + idx2] = local_node.update_weights(model=copy.deepcopy(local_models[idx1 * len(datasets_name) + idx2]), train_loader=train_loader_list[idx1 * len(datasets_name) + idx2])
        loss_temp = [0 for i in range(len(datasets_name))]
        acc_temp = [0 for i in range(len(datasets_name))]
        for idx1 in range(args.num_users // len(datasets_name)):
            with torch.no_grad():
                for idx2 in range(len(datasets_name)):
                    local_test = LocalTest(args=args)
                    loss, _ = local_test.test_inference(args, local_models[idx1 * len(datasets_name) + idx2], train_loader_list[idx1 * len(datasets_name) + idx2])
                    loss_temp[idx2] += loss
                    _, acc = local_test.test_inference(args, local_models[idx1 * len(datasets_name) + idx2], test_loader_list[idx2])
                    acc_temp[idx2] += acc
        for idx in range(len(datasets_name)):
            print('{:<11s} | train loss: {:.4f} | Test Acc: {:.4f}'.format(datasets_name[idx], loss_temp[idx] / (args.num_users // len(datasets_name)), acc_temp[idx] / (args.num_users // len(datasets_name))))
            train_loss[datasets_name[idx]].append(copy.deepcopy(loss_temp[idx] / (args.num_users // len(datasets_name))))
            accuracy_list[datasets_name[idx]].append(copy.deepcopy(acc_temp[idx] / (args.num_users // len(datasets_name))))
    return train_loss, accuracy_list, datasets_name

def fedavg(args, train_loader_list, test_loader_list):
    loader_size = [len(train_loader.dataset) for train_loader in train_loader_list]
    client_weights = [item / sum(loader_size) for item in loader_size]
    if args.dataset == "digit":
        global_model = adcol_model().to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
    elif args.dataset == "office":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["amazon", "caltech", "dslr", "webcam"]
    elif args.dataset == "PACS":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["art_painting", "cartoon", "photo", "sketch"]
    train_loss = {item: [] for item in datasets_name}
    accuracy_list = {item: [] for item in datasets_name}
    for round in tqdm(range(args.iters)):
        print(f'\n | Global Training Round : {round} |\n')
        # select clients
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                local_node = LocalUpdate(args=args)
                local_models[idx1 * len(datasets_name) + idx2] = local_node.update_weights(model=copy.deepcopy(local_models[idx1 * len(datasets_name) + idx2]), train_loader=train_loader_list[idx1 * len(datasets_name) + idx2])
        # update global model
        global_model, local_models = communication(args, copy.deepcopy(global_model), copy.deepcopy(local_models), client_weights)
        loss_temp = [0 for i in range(len(datasets_name))]
        acc_temp = [0 for i in range(len(datasets_name))]
        for idx1 in range(args.num_users // len(datasets_name)):
            with torch.no_grad():
                for idx2 in range(len(datasets_name)):
                    local_test = LocalTest(args=args)
                    loss, _ = local_test.test_inference(args, local_models[idx1 * len(datasets_name) + idx2], train_loader_list[idx1 * len(datasets_name) + idx2])
                    loss_temp[idx2] += loss
                    _, acc = local_test.test_inference(args, local_models[idx1 * len(datasets_name) + idx2], test_loader_list[idx2])
                    acc_temp[idx2] += acc
        for idx in range(len(datasets_name)):
            print('{:<11s} | train loss: {:.4f} | Test Acc: {:.4f}'.format(datasets_name[idx], loss_temp[idx] / (args.num_users // len(datasets_name)), acc_temp[idx] / (args.num_users // len(datasets_name))))
            train_loss[datasets_name[idx]].append(copy.deepcopy(loss_temp[idx] / (args.num_users // len(datasets_name))))
            accuracy_list[datasets_name[idx]].append(copy.deepcopy(acc_temp[idx] / (args.num_users // len(datasets_name))))
    return train_loss, accuracy_list, datasets_name

def fedProx(args, train_loader_list, test_loader_list):
    loader_size = [len(train_loader.dataset) for train_loader in train_loader_list]
    client_weights = [item / sum(loader_size) for item in loader_size]
    if args.dataset == "digit":
        global_model = adcol_model().to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
    elif args.dataset == "office":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["amazon", "caltech", "dslr", "webcam"]
    elif args.dataset == "PACS":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["art_painting", "cartoon", "photo", "sketch"]
    train_loss = {item: [] for item in datasets_name}
    accuracy_list = {item: [] for item in datasets_name}
    for round in tqdm(range(args.iters)):
        print(f'\n | Global Training Round : {round} |\n')
        # select clients
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                local_node = LocalUpdate(args=args)
                if round == 0:
                    local_models[idx1 * len(datasets_name) + idx2] = local_node.update_weights(model=copy.deepcopy(local_models[idx1 * len(datasets_name) + idx2]), train_loader=train_loader_list[idx1 * len(datasets_name) + idx2])
                else:
                    local_models[idx1 * len(datasets_name) + idx2] = local_node.update_weights_fedProx(model=copy.deepcopy(local_models[idx1 * len(datasets_name) + idx2]), train_loader=train_loader_list[idx1 * len(datasets_name) + idx2])
        # update global model
        global_model, local_models = communication(args, copy.deepcopy(global_model), copy.deepcopy(local_models), client_weights)
        loss_temp = [0 for i in range(len(datasets_name))]
        acc_temp = [0 for i in range(len(datasets_name))]
        for idx1 in range(args.num_users // len(datasets_name)):
            with torch.no_grad():
                for idx2 in range(len(datasets_name)):
                    local_test = LocalTest(args=args)
                    loss, _ = local_test.test_inference(args, local_models[idx1 * len(datasets_name) + idx2], train_loader_list[idx1 * len(datasets_name) + idx2])
                    loss_temp[idx2] += loss
                    _, acc = local_test.test_inference(args, local_models[idx1 * len(datasets_name) + idx2], test_loader_list[idx2])
                    acc_temp[idx2] += acc
        for idx in range(len(datasets_name)):
            print('{:<11s} | train loss: {:.4f} | Test Acc: {:.4f}'.format(datasets_name[idx], loss_temp[idx] / (args.num_users // len(datasets_name)), acc_temp[idx] / (args.num_users // len(datasets_name))))
            train_loss[datasets_name[idx]].append(copy.deepcopy(loss_temp[idx] / (args.num_users // len(datasets_name))))
            accuracy_list[datasets_name[idx]].append(copy.deepcopy(acc_temp[idx] / (args.num_users // len(datasets_name))))
    return train_loss, accuracy_list, datasets_name

def perfedavg(args, train_loader_list, test_loader_list):
    loader_size = [len(train_loader.dataset) for train_loader in train_loader_list]
    client_weights = [item / sum(loader_size) for item in loader_size]
    if args.dataset == "digit":
        global_model = adcol_model().to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
    elif args.dataset == "office":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["amazon", "caltech", "dslr", "webcam"]
    elif args.dataset == "PACS":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["art_painting", "cartoon", "photo", "sketch"]
    train_loss = {item: [] for item in datasets_name}
    accuracy_list = {item: [] for item in datasets_name}
    for round in tqdm(range(args.iters)):
        print(f'\n | Global Training Round : {round} |\n')
        # select clients
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                local_node = LocalUpdate(args=args)
                local_models[idx1 * len(datasets_name) + idx2] = local_node.update_weights_perfedavg(model=copy.deepcopy(local_models[idx1 * len(datasets_name) + idx2]), train_loader=train_loader_list[idx1 * len(datasets_name) + idx2])
        # update global model
        global_model, local_models = communication(args, copy.deepcopy(global_model), copy.deepcopy(local_models), client_weights)
        loss_temp = [0 for i in range(len(datasets_name))]
        acc_temp = [0 for i in range(len(datasets_name))]
        for idx1 in range(args.num_users // len(datasets_name)):
            with torch.no_grad():
                for idx2 in range(len(datasets_name)):
                    local_test = LocalTest(args=args)
                    loss, _ = local_test.test_inference(args, local_models[idx1 * len(datasets_name) + idx2], train_loader_list[idx1 * len(datasets_name) + idx2])
                    loss_temp[idx2] += loss
                    _, acc = local_test.test_inference(args, local_models[idx1 * len(datasets_name) + idx2], test_loader_list[idx2])
                    acc_temp[idx2] += acc
        for idx in range(len(datasets_name)):
            print('{:<11s} | train loss: {:.4f} | Test Acc: {:.4f}'.format(datasets_name[idx], loss_temp[idx] / (args.num_users // len(datasets_name)), acc_temp[idx] / (args.num_users // len(datasets_name))))
            train_loss[datasets_name[idx]].append(copy.deepcopy(loss_temp[idx] / (args.num_users // len(datasets_name))))
            accuracy_list[datasets_name[idx]].append(copy.deepcopy(acc_temp[idx] / (args.num_users // len(datasets_name))))
        
    return train_loss, accuracy_list, datasets_name

def fedrep(args, train_loader_list, test_loader_list):
    loader_size = [len(train_loader.dataset) for train_loader in train_loader_list]
    client_weights = [item / sum(loader_size) for item in loader_size]
    if args.dataset == "digit":
        global_model = adcol_model().to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
    elif args.dataset == "office":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["amazon", "caltech", "dslr", "webcam"]
    elif args.dataset == "PACS":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["art_painting", "cartoon", "photo", "sketch"]
    train_loss = {item: [] for item in datasets_name}
    accuracy_list = {item: [] for item in datasets_name}
    for round in tqdm(range(args.iters)):
        print(f'\n | Global Training Round : {round} |\n')
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                local_node = LocalUpdate(args=args)
                local_models[idx1 * len(datasets_name) + idx2] = local_node.update_weights_fedrep(model=copy.deepcopy(local_models[idx1 * len(datasets_name) + idx2]), train_loader=train_loader_list[idx1 * len(datasets_name) + idx2])
        # update global model
        global_model, local_models = communication(args, copy.deepcopy(global_model), copy.deepcopy(local_models), client_weights)
        loss_temp = [0 for i in range(len(datasets_name))]
        acc_temp = [0 for i in range(len(datasets_name))]
        for idx1 in range(args.num_users // len(datasets_name)):
            with torch.no_grad():
                for idx2 in range(len(datasets_name)):
                    local_test = LocalTest(args=args)
                    loss, _ = local_test.test_inference(args, local_models[idx1 * len(datasets_name) + idx2], train_loader_list[idx1 * len(datasets_name) + idx2])
                    loss_temp[idx2] += loss
                    _, acc = local_test.test_inference(args, local_models[idx1 * len(datasets_name) + idx2], test_loader_list[idx2])
                    acc_temp[idx2] += acc
        for idx in range(len(datasets_name)):
            print('{:<11s} | train loss: {:.4f} | Test Acc: {:.4f}'.format(datasets_name[idx], loss_temp[idx] / (args.num_users // len(datasets_name)), acc_temp[idx] / (args.num_users // len(datasets_name))))
            train_loss[datasets_name[idx]].append(copy.deepcopy(loss_temp[idx] / (args.num_users // len(datasets_name))))
            accuracy_list[datasets_name[idx]].append(copy.deepcopy(acc_temp[idx] / (args.num_users // len(datasets_name))))
        
    return train_loss, accuracy_list, datasets_name

def fedproto(args, train_loader_list, test_loader_list, num_list):
    loader_size = [len(train_loader.dataset) for train_loader in train_loader_list]
    client_weights = [item / sum(loader_size) for item in loader_size]
    if args.dataset == "digit":
        global_model = adcol_model().to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
    elif args.dataset == "office":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["amazon", "caltech", "dslr", "webcam"]
    elif args.dataset == "PACS":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["art_painting", "cartoon", "photo", "sketch"]
    train_loss = {item: [] for item in datasets_name}
    accuracy_list = {item: [] for item in datasets_name}
    global_proto = {}
    for round in tqdm(range(args.iters)):
        print(f'\n | Global Training Round : {round} |\n')
        # select clients
        protos = []
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                local_node = LocalUpdate(args=args)
                local_models[idx1 * len(datasets_name) + idx2], proto = local_node.update_weights_proto(model=copy.deepcopy(local_models[idx1 * len(datasets_name) + idx2]), train_loader=train_loader_list[idx1 * len(datasets_name) + idx2], global_proto=global_proto)
                protos.append(proto)
        # update global protos
        global_proto = proto_aggregation(protos, num_list)
        loss_temp = [0 for i in range(len(datasets_name))]
        acc_temp = [0 for i in range(len(datasets_name))]
        for idx1 in range(args.num_users // len(datasets_name)):
            with torch.no_grad():
                for idx2 in range(len(datasets_name)):
                    local_test = LocalTest(args=args)
                    loss, _ = local_test.test_inference_proto(args, local_models[idx1 * len(datasets_name) + idx2], train_loader_list[idx1 * len(datasets_name) + idx2], global_proto)
                    loss_temp[idx2] += loss
                    _, acc = local_test.test_inference_proto(args, local_models[idx1 * len(datasets_name) + idx2], test_loader_list[idx2], global_proto)
                    acc_temp[idx2] += acc
        for idx in range(len(datasets_name)):
            print('{:<11s} | train loss: {:.4f} | Test Acc: {:.4f}'.format(datasets_name[idx], loss_temp[idx] / (args.num_users // len(datasets_name)), acc_temp[idx] / (args.num_users // len(datasets_name))))
            train_loss[datasets_name[idx]].append(copy.deepcopy(loss_temp[idx] / (args.num_users // len(datasets_name))))
            accuracy_list[datasets_name[idx]].append(copy.deepcopy(acc_temp[idx] / (args.num_users // len(datasets_name))))
    return train_loss, accuracy_list, datasets_name

def fedBN(args, train_loader_list, test_loader_list):
    loader_size = [len(train_loader.dataset) for train_loader in train_loader_list]
    client_weights = [item / sum(loader_size) for item in loader_size]
    if args.dataset == "digit":
        global_model = adcol_model().to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
    elif args.dataset == "office":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["amazon", "caltech", "dslr", "webcam"]
    elif args.dataset == "PACS":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["art_painting", "cartoon", "photo", "sketch"]
    train_loss = {item: [] for item in datasets_name}
    accuracy_list = {item: [] for item in datasets_name}
    for round in tqdm(range(args.iters)):
        print(f'\n | Global Training Round : {round} |\n')
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                local_node = LocalUpdate(args=args)
                local_models[idx1 * len(datasets_name) + idx2]= local_node.update_weights(model=copy.deepcopy(local_models[idx1 * len(datasets_name) + idx2]), train_loader=train_loader_list[idx1 * len(datasets_name) + idx2])
        # update global model
        global_model, local_models = communication(args, copy.deepcopy(global_model), copy.deepcopy(local_models), client_weights)
        loss_temp = [0 for i in range(len(datasets_name))]
        acc_temp = [0 for i in range(len(datasets_name))]
        for idx1 in range(args.num_users // len(datasets_name)):
            with torch.no_grad():
                for idx2 in range(len(datasets_name)):
                    local_test = LocalTest(args=args)
                    loss, _ = local_test.test_inference(args, local_models[idx1 * len(datasets_name) + idx2], train_loader_list[idx1 * len(datasets_name) + idx2])
                    loss_temp[idx2] += loss
                    _, acc = local_test.test_inference(args, local_models[idx1 * len(datasets_name) + idx2], test_loader_list[idx2])
                    acc_temp[idx2] += acc
        for idx in range(len(datasets_name)):
            print('{:<11s} | train loss: {:.4f} | Test Acc: {:.4f}'.format(datasets_name[idx], loss_temp[idx] / (args.num_users // len(datasets_name)), acc_temp[idx] / (args.num_users // len(datasets_name))))
            train_loss[datasets_name[idx]].append(copy.deepcopy(loss_temp[idx] / (args.num_users // len(datasets_name))))
            accuracy_list[datasets_name[idx]].append(copy.deepcopy(acc_temp[idx] / (args.num_users // len(datasets_name))))
    return train_loss, accuracy_list, datasets_name

def moon(args, train_loader_list, test_loader_list):
    loader_size = [len(train_loader.dataset) for train_loader in train_loader_list]
    client_weights = [item / sum(loader_size) for item in loader_size]
    if args.dataset == "digit":
        global_model = adcol_model().to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
    elif args.dataset == "office":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["amazon", "caltech", "dslr", "webcam"]
    elif args.dataset == "PACS":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["art_painting", "cartoon", "photo", "sketch"]
    train_loss = {item: [] for item in datasets_name}
    accuracy_list = {item: [] for item in datasets_name}
    for round in tqdm(range(args.iters)):
        print(f'\n | Global Training Round : {round} |\n')
        # select clients
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                if round == 0:
                    local_node = LocalUpdate(args=args)
                    local_node.old_model = copy.deepcopy(local_models[idx1 * len(datasets_name) + idx2])
                old_model = copy.deepcopy(local_node.old_model)
                local_models[idx1 * len(datasets_name) + idx2] = local_node.update_weights_moon(model=copy.deepcopy(local_models[idx1 * len(datasets_name) + idx2]), train_loader=train_loader_list[idx1 * len(datasets_name) + idx2])
        # update global model
        global_model, local_models = communication(args, global_model, local_models, client_weights)
        loss_temp = [0 for i in range(len(datasets_name))]
        acc_temp = [0 for i in range(len(datasets_name))]
        for idx1 in range(args.num_users // len(datasets_name)):
            with torch.no_grad():
                for idx2 in range(len(datasets_name)):
                    local_test = LocalTest(args=args)
                    loss, _ = local_test.test_inference_moon(args, old_model, local_models[idx1 * len(datasets_name) + idx2], global_model, train_loader_list[idx1 * len(datasets_name) + idx2])
                    loss_temp[idx2] += loss
                    _, acc = local_test.test_inference_moon(args, old_model, local_models[idx1 * len(datasets_name) + idx2], global_model,  test_loader_list[idx2])
                    acc_temp[idx2] += acc
        for idx in range(len(datasets_name)):
            print('{:<11s} | train loss: {:.4f} | Test Acc: {:.4f}'.format(datasets_name[idx], loss_temp[idx] / (args.num_users // len(datasets_name)), acc_temp[idx] / (args.num_users // len(datasets_name))))
            train_loss[datasets_name[idx]].append(copy.deepcopy(loss_temp[idx] / (args.num_users // len(datasets_name))))
            accuracy_list[datasets_name[idx]].append(copy.deepcopy(acc_temp[idx] / (args.num_users // len(datasets_name))))
    return train_loss, accuracy_list, datasets_name

def adcol(args, train_loader_list, test_loader_list):
    loader_size = [len(train_loader.dataset) for train_loader in train_loader_list]
    client_weights = [item / sum(loader_size) for item in loader_size]
    if args.dataset == "digit":
        global_model = adcol_model().to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        discriminator = Discriminator(global_model, args.num_users).to(args.device)
        datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
    elif args.dataset == "office":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        discriminator = Discriminator(global_model, args.num_users).to(args.device)
        datasets_name = ["amazon", "caltech", "dslr", "webcam"]
    elif args.dataset == "PACS":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        discriminator = Discriminator(global_model, args.num_users).to(args.device)
        datasets_name = ["art_painting", "cartoon", "photo", "sketch"]
    train_loss = {item: [] for item in datasets_name}
    accuracy_list = {item: [] for item in datasets_name}
    discriminator_optimizer = torch.optim.SGD(discriminator.parameters(), lr=args.lr, weight_decay=1e-5, momentum=0.9)
    for round in tqdm(range(args.iters)):
        print(f'\n | Global Training Round : {round} |\n')
        # select clients
        features_labels = []
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                local_node = LocalUpdate(args=args)
                local_models[idx1 * len(datasets_name) + idx2], features = local_node.update_weights_adcol(model=copy.deepcopy(local_models[idx1 * len(datasets_name) + idx2]), discriminator=copy.deepcopy(discriminator), train_loader=train_loader_list[idx1 * len(datasets_name) + idx2])
                ids = torch.ones(features.shape[0]) * (idx1 * len(datasets_name) + idx2)
                features_labels.append([features, ids])
        loss_temp = [0 for i in range(len(datasets_name))]
        loss_kl = [0 for i in range(len(datasets_name))]
        acc_temp = [0 for i in range(len(datasets_name))]
        for idx1 in range(args.num_users // len(datasets_name)):
            with torch.no_grad():
                for idx2 in range(len(datasets_name)):
                    local_test = LocalTest(args=args)
                    loss, kl_loss,  _ = local_test.test_inference_adcol(args, local_models[idx1 * len(datasets_name) + idx2], train_loader_list[idx1 * len(datasets_name) + idx2], copy.deepcopy(discriminator))
                    loss_temp[idx2] += loss
                    loss_kl[idx2] += kl_loss
                    _, _, acc = local_test.test_inference_adcol(args, local_models[idx1 * len(datasets_name) + idx2], test_loader_list[idx2], copy.deepcopy(discriminator))
                    acc_temp[idx2] += acc
        for idx in range(len(datasets_name)):
            print('\n{:<11s} | train loss: {:.4f} | kl loss : {:.4f}| Test Acc: {:.4f}'.format(datasets_name[idx], loss_temp[idx] / (args.num_users // len(datasets_name)), loss_kl[idx] / (args.num_users // len(datasets_name)), acc_temp[idx] / (args.num_users // len(datasets_name))))
            train_loss[datasets_name[idx]].append(copy.deepcopy(loss_temp[idx] / (args.num_users // len(datasets_name))))
            accuracy_list[datasets_name[idx]].append(copy.deepcopy(acc_temp[idx] / (args.num_users // len(datasets_name))))
        # update discriminatro model
        features_dataset = data_utils.FeatureDataset(features_labels)
        features_loader = torch.utils.data.DataLoader(features_dataset, batch_size=args.batch, shuffle=True)  
        loss_func = nn.CrossEntropyLoss()
        discriminator.train()
        for _ in range(args.r_epoch):
            for x, y in features_loader:
                x, y = x.to(args.device).float(), y.to(args.device).long()
                y_pred = discriminator(x)
                loss = loss_func(y_pred, y).mean()
                discriminator_optimizer.zero_grad()
                loss.backward()
                discriminator_optimizer.step()
    return train_loss, accuracy_list, datasets_name

def RUCR(args, train_loader_list, test_loader_list):
    loader_size = [len(train_loader.dataset) for train_loader in train_loader_list]
    client_weights = [item / sum(loader_size) for item in loader_size]
    if args.dataset == "digit":
        global_model = adcol_model().to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
    elif args.dataset == "office":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["amazon", "caltech", "dslr", "webcam"]
    elif args.dataset == "PACS":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["art_painting", "cartoon", "photo", "sketch"]
    train_loss = {item: [] for item in datasets_name}
    accuracy_list = {item: [] for item in datasets_name}
    for round in tqdm(range(args.iters)):
        print(f'\n | Global Training Round : {round} |\n')
        protos, features = [], []
        num = []
        num_list_ = Counter()
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                local_node = LocalUpdate(args=args)
                avg_proto, feature, num_list = local_node.compute_proto(model=copy.deepcopy(local_models[idx1 * len(datasets_name) + idx2]), train_loader=train_loader_list[idx1 * len(datasets_name) + idx2])
                num_list_ += num_list
                num.append(num_list)
                features.append(feature)
                protos.append(avg_proto)
        ratio = get_cls_ratio(args, num_list_)
        global_proto = get_mean(args, protos, num)
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                local_models[idx1 * len(datasets_name) + idx2] = local_node.update_weights_rucr(model=copy.deepcopy(local_models[idx1 * len(datasets_name) + idx2]), train_loader=train_loader_list[idx1 * len(datasets_name) + idx2], global_proto=global_proto, ratio_list=ratio)
        # update global model by fedavg
        global_model, local_models = communication(args, global_model, local_models, client_weights)
        loss_temp = [0 for i in range(len(datasets_name))]
        acc_temp = [0 for i in range(len(datasets_name))]
        for idx1 in range(args.num_users // len(datasets_name)):
            with torch.no_grad():
                for idx2 in range(len(datasets_name)):
                    local_test = LocalTest(args=args)
                    loss, _ = local_test.test_inference(args, local_models[idx1 * len(datasets_name) + idx2], train_loader_list[idx1 * len(datasets_name) + idx2])
                    loss_temp[idx2] += loss
                    _, acc = local_test.test_inference(args, local_models[idx1 * len(datasets_name) + idx2], test_loader_list[idx2])
                    acc_temp[idx2] += acc
        for idx in range(len(datasets_name)):
            print('{:<11s} | train loss: {:.4f} | Test Acc: {:.4f}'.format(datasets_name[idx], loss_temp[idx] / (args.num_users // len(datasets_name)), acc_temp[idx] / (args.num_users // len(datasets_name))))
            train_loss[datasets_name[idx]].append(copy.deepcopy(loss_temp[idx] / (args.num_users // len(datasets_name))))
            accuracy_list[datasets_name[idx]].append(copy.deepcopy(acc_temp[idx] / (args.num_users // len(datasets_name))))
        # local training - classifier learning
        norm_means = cal_norm_mean(args, protos, num)
        mixup_cls_params = []
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                mixup_cls_param = local_node.local_crt(copy.deepcopy(local_models[idx1 * len(datasets_name) + idx2]), norm_means, features[idx1 * len(datasets_name) + idx2])
                mixup_cls_params.append(mixup_cls_param)
        mixup_classifier = model_fusion(mixup_cls_params, loader_size)
        global_model.classifier.load_state_dict(mixup_classifier)
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                local_models[idx1 * len(datasets_name) + idx2] = copy.deepcopy(global_model)
    return train_loss, accuracy_list, datasets_name
 
def ours(args, train_loader_list, test_loader_list):
    loader_size = [len(train_loader.dataset) for train_loader in train_loader_list]
    client_weights = [item / sum(loader_size) for item in loader_size]
    best_local_models = []
    if args.dataset == "digit":
        global_model = adcol_model().to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        discriminator = Discriminator(global_model, args.num_users).to(args.device)
        datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
    elif args.dataset == "office":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        discriminator = Discriminator(global_model, args.num_users).to(args.device)
        datasets_name = ["amazon", "caltech", "dslr", "webcam"]
    elif args.dataset == "PACS":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        discriminator = Discriminator(global_model, args.num_users).to(args.device)
        datasets_name = ["art_painting", "cartoon", "photo", "sketch"]
    global_classifier = classifier_model(args, args.num_classes).to(args.device)
    train_loss = {item: [] for item in datasets_name}
    accuracy_list = {item: [] for item in datasets_name}
    discriminator_optimizer = torch.optim.SGD(discriminator.parameters(), lr=args.lr, weight_decay=1e-5, momentum=0.9)
    classifier_optimizer = torch.optim.SGD(global_classifier.parameters(), lr=args.lr, weight_decay=1e-5, momentum=0.9)
    global_proto = {}
    best_acc = 0
    for round in tqdm(range(args.iters)):
        print(f'\n | Global Training Round : {round} |\n')
        # select clients
        features_idx = []
        features_labels = []
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                local_node = LocalUpdate(args=args)
                local_models[idx1 * len(datasets_name) + idx2], features_label= local_node.update_weights_ours(round, model=copy.deepcopy(local_models[idx1 * len(datasets_name) + idx2]), discriminator=copy.deepcopy(discriminator), classifier_model=copy.deepcopy(global_classifier), train_loader=train_loader_list[idx1 * len(datasets_name) + idx2],global_proto=global_proto, momentum = args.momentum)
                features_labels.append([features_label[0], features_label[1]])
        local_protos = []
        num = []
        num_list_ = Counter()
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                avg_proto, num_list = local_node.compute_proto_debug(train_loader=features_labels[idx1 * len(datasets_name) + idx2])
                local_protos.append(avg_proto)
                num_list_ += num_list
                num.append(num_list)
        # ratio = get_cls_ratio(args, num_list_)
        global_proto = get_mean(args, local_protos, num)
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                features = features_labels[idx1 * len(datasets_name) + idx2][0]
                labels = features_labels[idx1 * len(datasets_name) + idx2][1]
                size = features.shape[1]
                mask = generate_random_tensor(size, p=0.9).to(args.device)
                lam = np.round(args.uniform_left + args.uniform_right * np.random.random(), 2)
                features_noise = []
                for idx, label in enumerate(labels):
                    features_noise.append(lam * features[idx] + (1 - lam) * global_proto[label.item()])
                features_noise = torch.stack(features_noise, dim=0)
                features_masked = features_noise * mask
                features_labels[idx1 * len(datasets_name) + idx2] = [features_masked, labels]
                ids = torch.ones(features.shape[0]) * (idx1 * len(datasets_name) + idx2)
                features_idx.append([features_masked, ids])
        loss_temp = [0 for i in range(len(datasets_name))]
        loss_kl = [0 for i in range(len(datasets_name))]
        acc_temp = [0 for i in range(len(datasets_name))]
        for idx1 in range(args.num_users // len(datasets_name)):
            with torch.no_grad():
                for idx2 in range(len(datasets_name)):
                    local_test = LocalTest(args=args)
                    loss, kl_loss,  _ = local_test.test_inference_ours(args, local_models[idx1 * len(datasets_name) + idx2], train_loader_list[idx1 * len(datasets_name) + idx2], copy.deepcopy(discriminator))
                    loss_temp[idx2] += loss
                    loss_kl[idx2] += kl_loss
                    _, _, acc = local_test.test_inference_ours(args, local_models[idx1 * len(datasets_name) + idx2], test_loader_list[idx2], copy.deepcopy(discriminator))
                    acc_temp[idx2] += acc
        for idx in range(len(datasets_name)):
            print('\n{:<11s} | train loss: {:.4f} | kl loss : {:.4f}| Test Acc: {:.4f}'.format(datasets_name[idx], loss_temp[idx] / (args.num_users // len(datasets_name)), loss_kl[idx] / (args.num_users // len(datasets_name)), acc_temp[idx] / (args.num_users // len(datasets_name))))
            train_loss[datasets_name[idx]].append(copy.deepcopy(loss_temp[idx] / (args.num_users // len(datasets_name))))
            accuracy_list[datasets_name[idx]].append(copy.deepcopy(acc_temp[idx] / (args.num_users // len(datasets_name))))
        # if sum(acc_temp) / len(acc_temp) > best_acc:
        #     for idx, dataset in enumerate(datasets_name):
        #         model_paths = f"weights/{args.seed}/{args.dataset}/"
        #         os.makedirs(model_paths, exist_ok=True)
        #         model_save_path = f"best_local_model_{dataset}.pth"
        #         torch.save(local_models[idx].state_dict(), model_paths + model_save_path)
        #     best_acc = sum(acc_temp) / len(acc_temp)
        # update global classifier layer
        features_label_dataset = data_utils.FeatureDataset(features_labels)
        features_label_loader = torch.utils.data.DataLoader(features_label_dataset, batch_size=args.batch, shuffle=True)  
        loss_func = nn.CrossEntropyLoss()
        global_classifier.train()
        for _ in range(args.r_epoch):
            for x, y in features_label_loader:
                x, y= x.to(args.device).float(), y.to(args.device).long()
                y_pred = global_classifier(x)
                loss1 = loss_func(y_pred, y).mean()
                classifier_optimizer.zero_grad()
                loss1.backward()
                classifier_optimizer.step()
        # update discriminator model
        features_dataset = data_utils.FeatureDataset(features_idx)
        features_loader = torch.utils.data.DataLoader(features_dataset, batch_size=args.batch, shuffle=True)  
        discriminator.train()
        for _ in range(args.r_epoch):
            for x, y in features_loader:
                x, y = x.to(args.device).float(), y.to(args.device).long()
                y_pred = discriminator(x)
                loss2 = loss_func(y_pred, y).mean()
                discriminator_optimizer.zero_grad()
                loss2.backward()
                discriminator_optimizer.step()
    return train_loss, accuracy_list, datasets_name

def ablation1(args, train_loader_list, test_loader_list):
    loader_size = [len(train_loader.dataset) for train_loader in train_loader_list]
    client_weights = [item / sum(loader_size) for item in loader_size]
    if args.dataset == "digit":
        global_model = adcol_model().to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        discriminator = Discriminator(global_model, args.num_users).to(args.device)
        datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
    elif args.dataset == "office":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        discriminator = Discriminator(global_model, args.num_users).to(args.device)
        datasets_name = ["amazon", "caltech", "dslr", "webcam"]
    elif args.dataset == "PACS":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        discriminator = Discriminator(global_model, args.num_users).to(args.device)
        datasets_name = ["art_painting", "cartoon", "photo", "sketch"]
    global_classifier = classifier_model(args, args.num_classes).to(args.device)
    train_loss = {item: [] for item in datasets_name}
    accuracy_list = {item: [] for item in datasets_name}
    discriminator_optimizer = torch.optim.SGD(discriminator.parameters(), lr=args.lr, weight_decay=1e-5, momentum=0.9)
    classifier_optimizer = torch.optim.SGD(global_classifier.parameters(), lr=args.lr, weight_decay=1e-5, momentum=0.9)
    global_proto = {}
    best_acc = 0
    for round in tqdm(range(args.iters)):
        print(f'\n | Global Training Round : {round} |\n')
        # select clients
        features_idx = []
        features_labels = [[] for i in range(len(datasets_name))]
        features_origin = []
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                local_node = LocalUpdate(args=args)
                local_models[idx1 * len(datasets_name) + idx2], features_label= local_node.update_weights_ours_ablation1(round, model=copy.deepcopy(local_models[idx1 * len(datasets_name) + idx2]), discriminator=copy.deepcopy(discriminator), classifier_model=copy.deepcopy(global_classifier), train_loader=train_loader_list[idx1 * len(datasets_name) + idx2],global_proto=global_proto, momentum = args.momentum)
                features_origin.append([features_label[0], features_label[1]])
        local_protos = []
        num = []
        num_list_ = Counter()
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                avg_proto, num_list = local_node.compute_proto_debug(train_loader=features_origin[idx1 * len(datasets_name) + idx2])
                local_protos.append(avg_proto)
                num_list_ += num_list
                num.append(num_list)
        ## must
        global_proto = get_mean(args, local_protos, num)
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                features = features_origin[idx1 * len(datasets_name) + idx2][0]
                labels = features_origin[idx1 * len(datasets_name) + idx2][1]
                size = features.shape[1]
                mask = generate_random_tensor(size, p=0.9).to(args.device)
                lam = np.round(args.uniform_left + args.uniform_right * np.random.random(), 2)
                features_noise = []
                for idx, label in enumerate(labels):
                    features_noise.append(lam * features[idx] + (1 - lam) * global_proto[label.item()])
                features_noise = torch.stack(features_noise, dim=0)
                features_masked = features_noise * mask
                features_labels[idx1 * len(datasets_name) + idx2] = [features_masked, labels]
                ids = torch.ones(features.shape[0]) * (idx1 * len(datasets_name) + idx2)
                features_idx.append([features_masked, ids])
        loss_temp = [0 for i in range(len(datasets_name))]
        loss_kl = [0 for i in range(len(datasets_name))]
        acc_temp = [0 for i in range(len(datasets_name))]
        loss_info = [0 for i in range(len(datasets_name))]
        for idx1 in range(args.num_users // len(datasets_name)):
            with torch.no_grad():
                for idx2 in range(len(datasets_name)):
                    local_test = LocalTest(args=args)
                    loss, kl_loss, info_loss, _ = local_test.test_inference_ablation1(args, local_models[idx1 * len(datasets_name) + idx2], train_loader_list[idx1 * len(datasets_name) + idx2], global_proto=global_proto, dmodel = copy.deepcopy(discriminator))
                    loss_temp[idx2] += loss
                    loss_kl[idx2] += kl_loss
                    loss_info[idx2] += info_loss
                    _, _, _, acc = local_test.test_inference_ablation1(args, local_models[idx1 * len(datasets_name) + idx2], test_loader_list[idx2], global_proto=global_proto, dmodel = copy.deepcopy(discriminator))
                    acc_temp[idx2] += acc
        for idx in range(len(datasets_name)):
            print('\n{:<11s} | CE loss: {:.4f} | kl loss : {:.4f}| infoNCEloss : {:.4f}| Test Acc: {:.4f}'.format(datasets_name[idx], loss_temp[idx] / (args.num_users // len(datasets_name)), loss_kl[idx] / (args.num_users // len(datasets_name)), loss_info[idx] / (args.num_users // len(datasets_name)),  acc_temp[idx] / (args.num_users // len(datasets_name))))
            train_loss[datasets_name[idx]].append(copy.deepcopy(loss_temp[idx] / (args.num_users // len(datasets_name))))
            accuracy_list[datasets_name[idx]].append(copy.deepcopy(acc_temp[idx] / (args.num_users // len(datasets_name))))
        if np.average(acc_temp) > best_acc:
            file_path = f'pkl/{args.dataset}/{args.mode}_{args.seed}_features_labels_idx_loss{args.loss_component}.pkl'
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            for i in range(args.num_users):
                features_origin[i][0] = features_origin[i][0].to("cpu")
                features_origin[i][1] = features_origin[i][1].to("cpu")
            with open(file_path, 'wb') as f:
                pickle.dump(features_origin, f)
            best_acc = np.average(acc_temp) 
        
        # update global classifier layer
        features_label_dataset = data_utils.FeatureDataset(features_labels)
        features_label_loader = torch.utils.data.DataLoader(features_label_dataset, batch_size=args.batch, shuffle=True)  
        loss_func = nn.CrossEntropyLoss()
        global_classifier.train()
        for _ in range(args.r_epoch):
            for x, y in features_label_loader:
                x, y= x.to(args.device).float(), y.to(args.device).long()
                y_pred = global_classifier(x)
                loss1 = loss_func(y_pred, y).mean()
                classifier_optimizer.zero_grad()
                loss1.backward()
                classifier_optimizer.step()
        # update discriminator model
        if args.loss_component in [3, 4]:
            features_dataset = data_utils.FeatureDataset(features_idx)
            features_loader = torch.utils.data.DataLoader(features_dataset, batch_size=args.batch, shuffle=True)  
            discriminator.train()
            for _ in range(args.r_epoch):
                for x, y in features_loader:
                    x, y = x.to(args.device).float(), y.to(args.device).long()
                    y_pred = discriminator(x)
                    loss2 = loss_func(y_pred, y).mean()
                    discriminator_optimizer.zero_grad()
                    loss2.backward()
                    discriminator_optimizer.step()
    return train_loss, accuracy_list, datasets_name

def ablation2(args, train_loader_list, test_loader_list):
    loader_size = [len(train_loader.dataset) for train_loader in train_loader_list]
    client_weights = [item / sum(loader_size) for item in loader_size]
    if args.dataset == "digit":
        global_model = adcol_model().to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        discriminator = Discriminator(global_model, args.num_users).to(args.device)
        datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
    elif args.dataset == "office":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        discriminator = Discriminator(global_model, args.num_users).to(args.device)
        datasets_name = ["amazon", "caltech", "dslr", "webcam"]
    elif args.dataset == "PACS":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        discriminator = Discriminator(global_model, args.num_users).to(args.device)
        datasets_name = ["art_painting", "cartoon", "photo", "sketch"]
    global_classifier = classifier_model(args, args.num_classes).to(args.device)
    train_loss = {item: [] for item in datasets_name}
    accuracy_list = {item: [] for item in datasets_name}
    discriminator_optimizer = torch.optim.SGD(discriminator.parameters(), lr=args.lr, weight_decay=1e-5, momentum=0.9)
    classifier_optimizer = torch.optim.SGD(global_classifier.parameters(), lr=args.lr, weight_decay=1e-5, momentum=0.9)
    global_proto = {}
    for round in tqdm(range(args.iters)):
        print(f'\n | Global Training Round : {round} |\n')
        # select clients
        features_idx = []
        features_labels = []
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                local_node = LocalUpdate(args=args)
                local_models[idx1 * len(datasets_name) + idx2], features_label= local_node.update_weights_ours_ablation2(round, model=copy.deepcopy(local_models[idx1 * len(datasets_name) + idx2]), discriminator=copy.deepcopy(discriminator), classifier_model=copy.deepcopy(global_classifier), train_loader=train_loader_list[idx1 * len(datasets_name) + idx2],global_proto=global_proto, momentum = args.momentum)
                features_labels.append([features_label[0], features_label[1]])
        local_protos = []
        num = []
        num_list_ = Counter()
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                avg_proto, num_list = local_node.compute_proto_debug(train_loader=features_labels[idx1 * len(datasets_name) + idx2])
                local_protos.append(avg_proto)
                num_list_ += num_list
                num.append(num_list)
        # ratio = get_cls_ratio(args, num_list_)
        global_proto = get_mean(args, local_protos, num)
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                features = features_labels[idx1 * len(datasets_name) + idx2][0]
                labels = features_labels[idx1 * len(datasets_name) + idx2][1]
                size = features.shape[1]
                mask = generate_random_tensor(size, p=0.9).to(args.device)
                lam = np.round(args.uniform_left + args.uniform_right * np.random.random(), 2)
                features_noise = []
                for idx, label in enumerate(labels):
                    features_noise.append(lam * features[idx] + (1 - lam) * global_proto[label.item()])
                features_noise = torch.stack(features_noise, dim=0)
                features_masked = features_noise * mask
                features_labels[idx1 * len(datasets_name) + idx2] = [features_masked, labels]
                ids = torch.ones(features.shape[0]) * (idx1 * len(datasets_name) + idx2)
                features_idx.append([features_masked, ids])
        loss_temp = [0 for i in range(len(datasets_name))]
        loss_kl = [0 for i in range(len(datasets_name))]
        acc_temp = [0 for i in range(len(datasets_name))]
        loss_info = [0 for i in range(len(datasets_name))]
        for idx1 in range(args.num_users // len(datasets_name)):
            with torch.no_grad():
                for idx2 in range(len(datasets_name)):
                    local_test = LocalTest(args=args)
                    loss, kl_loss, info_loss, _ = local_test.test_inference_ablation2(args, local_models[idx1 * len(datasets_name) + idx2], train_loader_list[idx1 * len(datasets_name) + idx2], global_proto=global_proto, dmodel = copy.deepcopy(discriminator))
                    loss_temp[idx2] += loss
                    loss_kl[idx2] += kl_loss
                    loss_info[idx2] += info_loss
                    _, _, _, acc = local_test.test_inference_ablation2(args, local_models[idx1 * len(datasets_name) + idx2], test_loader_list[idx2], global_proto=global_proto, dmodel = copy.deepcopy(discriminator))
                    acc_temp[idx2] += acc
        for idx in range(len(datasets_name)):
            print('\n{:<11s} | CE loss: {:.4f} | kl loss : {:.4f}| infoNCEloss : {:.4f}| Test Acc: {:.4f}'.format(datasets_name[idx], loss_temp[idx] / (args.num_users // len(datasets_name)), loss_kl[idx] / (args.num_users // len(datasets_name)), loss_info[idx] / (args.num_users // len(datasets_name)),  acc_temp[idx] / (args.num_users // len(datasets_name))))
            train_loss[datasets_name[idx]].append(copy.deepcopy(loss_temp[idx] / (args.num_users // len(datasets_name))))
            accuracy_list[datasets_name[idx]].append(copy.deepcopy(acc_temp[idx] / (args.num_users // len(datasets_name))))
        # update global classifier layer
        loss_func = nn.CrossEntropyLoss()
        if args.cls_component == 1:
            features_label_dataset = data_utils.FeatureDataset(features_labels)
            features_label_loader = torch.utils.data.DataLoader(features_label_dataset, batch_size=args.batch, shuffle=True)  
            global_classifier.train()
            for _ in range(args.r_epoch):
                for x, y in features_label_loader:
                    x, y= x.to(args.device).float(), y.to(args.device).long()
                    y_pred = global_classifier(x)
                    loss1 = loss_func(y_pred, y).mean()
                    classifier_optimizer.zero_grad()
                    loss1.backward()
                    classifier_optimizer.step()
        # update discriminator model
        features_dataset = data_utils.FeatureDataset(features_idx)
        features_loader = torch.utils.data.DataLoader(features_dataset, batch_size=args.batch, shuffle=True)  
        discriminator.train()
        for _ in range(args.r_epoch):
            for x, y in features_loader:
                x, y = x.to(args.device).float(), y.to(args.device).long()
                y_pred = discriminator(x)
                loss2 = loss_func(y_pred, y).mean()
                discriminator_optimizer.zero_grad()
                loss2.backward()
                discriminator_optimizer.step()
    return train_loss, accuracy_list, datasets_name

def set_seed(args):
    random.seed(args.seed)  
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args.device = args.device if torch.cuda.is_available() else 'cpu'
    
def fed_main(args):
    torch.cuda.set_device(args.device)
    if args.exp == 1:
        # main experiment
        # modes = ["SingleSet", "fedavg", "fedprox", "perfedavg", "fedrep", "fedproto", "fedBN", "moon", "adcol", "RUCR", "ours"]
        seeds = [0,1,2]
        args.save_path = f"../result/exp{args.exp}/{args.dataset}/resnet50/"
        os.makedirs(args.save_path, exist_ok=True)
        for seed in seeds:
            data = collections.defaultdict(list)
            args.seed = seed
            set_seed(args)
            if args.dataset == "office":
                args.num_classes = 10
                args.num_users = 4
                args.size = 64
                args.wk_iters = 10
                args.r_epoch = 3
                args.pall_mu = 0.1
                args.pall_beta = 0.1
                datasets_name = ["amazon", "caltech", "dslr", "webcam"]
                train_loader_list, test_loader_list = prepare_data_office_feature_noniid(args=args)
                num_list = compute_num_list(args, train_loader_list)
            elif args.dataset == "digit":
                args.num_classes = 10
                args.num_users = 5
                args.size = 28
                args.wk_iters = 5
                args.pall_mu = 0.7
                args.pall_beta = 0.3
                args.r_epoch = 1
                datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
                train_loader_list, test_loader_list = prepare_data_digit_feature_noniid(args=args)
                num_list = compute_num_list(args, train_loader_list)
            elif args.dataset == "PACS":
                args.num_classes = 7
                args.num_users = 4
                args.size = 64
                args.wk_iters = 10
                args.pall_mu = 0.1
                args.pall_beta = 0.1
                args.r_epoch = 3
                datasets_name = ["art_painting", "cartoon", "photo", "sketch"]
                train_loader_list, test_loader_list = prepare_data_PACS_feature_noniid(args=args)
                num_list = compute_num_list(args, train_loader_list)
            print(args)
            save_path = args.save_path + f"{args.mode}"
            os.makedirs(save_path, exist_ok=True)
            excel_file = save_path + "/test_acc.xlsx"
            if not os.path.exists(excel_file):
                pd.DataFrame({"dataset": datasets_name}).to_excel(excel_file, index=False)
            if args.mode == "fedBN":
                train_loss, accuracy_list, datasets_name = fedBN(args, train_loader_list, test_loader_list)
            elif args.mode == "fedavg":
                train_loss, accuracy_list, datasets_name = fedavg(args, train_loader_list, test_loader_list)
            elif args.mode == "SingleSet":
                train_loss, accuracy_list, datasets_name = SingleSet(args, train_loader_list, test_loader_list)
            elif args.mode == "fedprox":
                train_loss, accuracy_list, datasets_name = fedProx(args, train_loader_list, test_loader_list)
            elif args.mode == "fedproto":
                train_loss, accuracy_list, datasets_name = fedproto(args, train_loader_list, test_loader_list, num_list)
            elif args.mode == "fedrep":
                train_loss, accuracy_list, datasets_name = fedrep(args, train_loader_list, test_loader_list)
            elif args.mode == "perfedavg":
                train_loss, accuracy_list, datasets_name = perfedavg(args, train_loader_list, test_loader_list)
            elif args.mode == "moon":
                train_loss, accuracy_list, datasets_name = moon(args, train_loader_list, test_loader_list)
            elif args.mode == "adcol":
                train_loss, accuracy_list, datasets_name = adcol(args, train_loader_list, test_loader_list)
            elif args.mode == "ours":
                train_loss, accuracy_list, datasets_name = ours(args, train_loader_list, test_loader_list)
            test_avg_acc = []
            for idx in range(len(accuracy_list[datasets_name[0]])):
                test_avg_acc.append(sum([accuracy_list[dataset][idx] for dataset in datasets_name]) / 5) 
            max_value = np.max(test_avg_acc)
            indices = np.where(np.array(test_avg_acc) == max_value)[0]
            best_step = indices[-1]
            for dataset_name in datasets_name:
                data[f"seed:{seed}"].append(accuracy_list[dataset_name][best_step])
            with pd.ExcelWriter(excel_file, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                existing_data = pd.read_excel(excel_file)
                new_data = pd.DataFrame(data)
                combined_data = pd.concat([existing_data, new_data], axis=1)
                combined_data.to_excel(writer, index=False)
    elif args.exp == 2:
        # ablation1 Combination of different loss functions
        # 1) CE
        # 2) CE + infoNCE
        # 3) CE + KL
        # 4) CE + infoNCE + KL
        args.mode = "ablation1"
        args.seed = 0
        combinations = [1,2,3,4]
        args.save_path = f"../result/exp{args.exp}/{args.dataset}/"
        os.makedirs(args.save_path, exist_ok=True)
        for combination in combinations:
            args.loss_component = combination
            data = collections.defaultdict(list)
            set_seed(args)
            if args.dataset == "office":
                args.num_classes = 10
                args.num_users = 4
                args.size = 64
                args.wk_iters = 10
                args.r_epoch = 3
                args.pall_mu = 0.1
                args.pall_beta = 0.1
                datasets_name = ["amazon", "caltech", "dslr", "webcam"]
                train_loader_list, test_loader_list = prepare_data_office_feature_noniid(args=args)
                num_list = compute_num_list(args, train_loader_list)
            elif args.dataset == "digit":
                args.num_classes = 10
                args.num_users = 5
                args.size = 28
                args.wk_iters = 5
                args.pall_mu = 0.5
                args.pall_beta = 0.1
                args.r_epoch = 1
                datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
                train_loader_list, test_loader_list = prepare_data_digit_feature_noniid(args=args)
                num_list = compute_num_list(args, train_loader_list)
            elif args.dataset == "PACS":
                args.num_classes = 7
                args.num_users = 4
                args.size = 64
                args.wk_iters = 10
                args.pall_mu = 0.1
                args.pall_beta = 0.1
                args.r_epoch = 3
                datasets_name = ["art_painting", "cartoon", "photo", "sketch"]
                train_loader_list, test_loader_list = prepare_data_PACS_feature_noniid(args=args)
                num_list = compute_num_list(args, train_loader_list)
            print(args)
            save_path = args.save_path + f"loss_component_{args.loss_component}"
            os.makedirs(save_path, exist_ok=True)
            excel_file = save_path + "/test_acc.xlsx"
            if not os.path.exists(excel_file):
                pd.DataFrame({"dataset": datasets_name}).to_excel(excel_file, index=False)
            if args.mode == "ablation1":
                train_loss, accuracy_list, datasets_name = ablation1(args, train_loader_list, test_loader_list)
            test_avg_acc = []
            for idx in range(len(accuracy_list[datasets_name[0]])):
                test_avg_acc.append(sum([accuracy_list[dataset][idx] for dataset in datasets_name]) / 5) 
            max_value = np.max(test_avg_acc)
            indices = np.where(np.array(test_avg_acc) == max_value)[0]
            best_step = indices[-1]
            for dataset_name in datasets_name:
                data[f"seed:{args.seed}"].append(accuracy_list[dataset_name][best_step])
            with pd.ExcelWriter(excel_file, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                existing_data = pd.read_excel(excel_file)
                new_data = pd.DataFrame(data)
                combined_data = pd.concat([existing_data, new_data], axis=1)
                combined_data.to_excel(writer, index=False)
    elif args.exp == 3:
        # ablation2 Whether to replace the local classifier
        # 1) yes
        # 2) no
        args.mode = "ablation2"
        args.seed = 0
        cls_modes = [1,2]
        args.save_path = f"../result/exp{args.exp}/{args.dataset}/"
        os.makedirs(args.save_path, exist_ok=True)
        for cls_mode in cls_modes:
            args.cls_component = cls_mode
            data = collections.defaultdict(list)
            set_seed(args)
            if args.dataset == "office":
                args.num_classes = 10
                args.num_users = 4
                args.size = 64
                args.wk_iters = 10
                args.r_epoch = 3
                args.pall_mu = 0.1
                args.pall_beta = 0.1
                datasets_name = ["amazon", "caltech", "dslr", "webcam"]
                train_loader_list, test_loader_list = prepare_data_office_feature_noniid(args=args)
                num_list = compute_num_list(args, train_loader_list)
            elif args.dataset == "digit":
                args.num_classes = 10
                args.num_users = 5
                args.size = 28
                args.wk_iters = 5
                args.pall_mu = 0.5
                args.pall_beta = 0.1
                args.r_epoch = 1
                datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
                train_loader_list, test_loader_list = prepare_data_digit_feature_noniid(args=args)
                num_list = compute_num_list(args, train_loader_list)
            elif args.dataset == "PACS":
                args.num_classes = 7
                args.num_users = 4
                args.size = 64
                args.wk_iters = 10
                args.pall_mu = 0.1
                args.pall_beta = 0.1
                args.r_epoch = 3
                datasets_name = ["art_painting", "cartoon", "photo", "sketch"]
                train_loader_list, test_loader_list = prepare_data_PACS_feature_noniid(args=args)
                num_list = compute_num_list(args, train_loader_list)
            print(args)
            save_path = args.save_path + f"cls_strategy_{args.cls_component}"
            os.makedirs(save_path, exist_ok=True)
            excel_file = save_path + "/test_acc.xlsx"
            if not os.path.exists(excel_file):
                pd.DataFrame({"dataset": datasets_name}).to_excel(excel_file, index=False)
            if args.mode == "ablation2":
                train_loss, accuracy_list, datasets_name = ablation2(args, train_loader_list, test_loader_list)
            test_avg_acc = []
            for idx in range(len(accuracy_list[datasets_name[0]])):
                test_avg_acc.append(sum([accuracy_list[dataset][idx] for dataset in datasets_name]) / 5) 
            max_value = np.max(test_avg_acc)
            indices = np.where(np.array(test_avg_acc) == max_value)[0]
            best_step = indices[-1]
            for dataset_name in datasets_name:
                data[f"seed:{args.seed}"].append(accuracy_list[dataset_name][best_step])
            with pd.ExcelWriter(excel_file, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                existing_data = pd.read_excel(excel_file)
                new_data = pd.DataFrame(data)
                combined_data = pd.concat([existing_data, new_data], axis=1)
                combined_data.to_excel(writer, index=False)

if __name__ == "__main__":
    args = args_parser()
    args.seed = 1
    fed_main(args)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    