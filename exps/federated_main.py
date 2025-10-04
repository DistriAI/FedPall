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

def generate_random_tensor(size, p):
    assert 0 <= p <= 1, "p should be between 0 and 1"
    num_ones = int(size * p)
    tensor = torch.cat((torch.ones(num_ones), torch.zeros(size - num_ones)))
    tensor = tensor[torch.randperm(tensor.size(0))]
    return tensor.view(size)

def calculate_q(args, L, diff_dict_flatten, round, client_weights):
    M, G = args.num_users, diff_dict_flatten.shape[1]
    t = round
    indicator = (diff_dict_flatten >= 0).float()
    L = (L * (t - 1) + indicator) / t

    C = torch.where(diff_dict_flatten >= 0, L, 1 - L)  # c_{m,i} shape: (M, G)

    mask = (C >= args.tau).float()  # I(c_{m,i} >= τ)
    masked_diffs = mask * diff_dict_flatten
    d_m = (masked_diffs ** 2).sum(dim=1)  # d_m: (M,)

    if round == 1:
        args.p_t = torch.tensor(client_weights).to(args.device)
        args.delta_p_t = torch.zeros(M).to(args.device)

    d_sum = d_m.sum()
    delta_p_t = (1 - args.beta) * args.delta_p_t + args.beta * (d_m / d_sum)
    p_t = args.p_t + delta_p_t
    p_t = p_t / p_t.sum()

    args.p_t = p_t.detach()
    args.delta_p_t = delta_p_t.detach()

    p_t_expanded = p_t.view(M, 1)
    numerator = mask * p_t_expanded
    denominator = numerator.sum(dim=0, keepdim=True) 
    denominator_safe = denominator + (denominator == 0).float()
    # q_t: shape (M, G)
    q_t = numerator / denominator_safe

    return q_t, L

def compute_num_list(args, train_loader_list):
    num_list = [Counter() for idx in range(args.num_users)]
    for idx in range(args.num_users):
        train_set = iter(train_loader_list[idx])
        for batch_idx in range(len(train_set)):
            images, labels = next(train_set)
            images, labels = images.to(args.device).float(), labels.to(args.device).long()
            for label in labels:
                if label.item() not in num_list[idx]:
                    num_list[idx][label.item()] = 1
                else:
                    num_list[idx][label.item()]+=1
    return num_list

def model_fusion(list_dicts_local_params: list, list_nums_local_data: list):
    # fedavg
    local_params = copy.deepcopy(list_dicts_local_params[0])
    for name_param in list_dicts_local_params[0]:
        list_values_param = []
        for dict_local_params, num_local_data in zip(list_dicts_local_params, list_nums_local_data):
            list_values_param.append(dict_local_params[name_param] * num_local_data)
        value_global_param = sum(list_values_param) / sum(list_nums_local_data)
        local_params[name_param] = value_global_param
    return local_params

def communication(args, server_model, models, client_weights):
    with torch.no_grad():
        # aggregate params
        if args.mode.lower() == 'fedbn':
            for key in server_model.state_dict().keys():
                if 'bn' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(args.num_users):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(args.num_users):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        elif args.mode.lower() == 'fedrep':
            for key in server_model.features.state_dict().keys():
                temp = torch.zeros_like(server_model.features.state_dict()[key], dtype=torch.float32)
                for client_idx in range(args.num_users):
                    temp += client_weights[client_idx] * models[client_idx].features.state_dict()[key]
                server_model.features.state_dict()[key].data.copy_(temp)
                for client_idx in range(args.num_users):
                    models[client_idx].features.state_dict()[key].data.copy_(server_model.features.state_dict()[key])
        else:
            for key in server_model.state_dict().keys():
                # num_batches_tracked is a non trainable LongTensor and
                # num_batches_tracked are the same for all clients for the given datasets
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return copy.deepcopy(server_model), copy.deepcopy(models)

def communication_heal(args, server_model, diff_dict, q):
    M, G = diff_dict.shape
    weighted_update_flat = (q * diff_dict).sum(dim=0)  # shape (G,)
    idx = 0
    with torch.no_grad():
        for param in server_model.parameters():
            if param.requires_grad:
                num_param = param.numel()
                update_chunk = weighted_update_flat[idx:idx + num_param].view_as(param).to(args.device)
                param.add_(update_chunk)
                idx += num_param

    return copy.deepcopy(server_model)

def get_cls_ratio(args, num_list):
    total_count = sum(num_list.values())
    proportion_list = {key:num / total_count for key, num in num_list.items()}
    return proportion_list

def cal_norm_mean(args, c_means, c_dis):
    glo_means = dict()
    c_dis_temp = torch.ones((args.num_users, args.num_classes))
    for idx in range(args.num_users):
        for key, value in c_dis[idx].items():
            c_dis_temp[idx][key] = value
    c_dis = c_dis_temp.to(args.device)
    total_num_per_cls = c_dis.sum(dim=0)
    for i in range(args.num_classes):
        for c_idx, c_mean in enumerate(c_means):
            if i not in c_mean.keys():
                continue
            temp = glo_means.get(i, 0)
            # normalize the local prototypes, send the direction to the server
            glo_means[i] = temp + \
                F.normalize(c_mean[i].view(1, -1),
                            dim=1).view(-1) * c_dis[c_idx][i]
        if glo_means.get(i) == None:
            continue
        t = glo_means[i]
        glo_means[i] = t / total_num_per_cls[i]
    return glo_means

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
                    loss, _ = local_test.test_inference(args, global_model, train_loader_list[idx1 * len(datasets_name) + idx2])
                    loss_temp[idx2] += loss
                    _, acc = local_test.test_inference(args, global_model, test_loader_list[idx2])
                    acc_temp[idx2] += acc
        for idx in range(len(datasets_name)):
            print('{:<11s} | train loss: {:.4f} | Test Acc: {:.4f}'.format(datasets_name[idx], loss_temp[idx] / (args.num_users // len(datasets_name)), acc_temp[idx] / (args.num_users // len(datasets_name))))
            train_loss[datasets_name[idx]].append(copy.deepcopy(loss_temp[idx] / (args.num_users // len(datasets_name))))
            accuracy_list[datasets_name[idx]].append(copy.deepcopy(acc_temp[idx] / (args.num_users // len(datasets_name))))
    return train_loss, accuracy_list, datasets_name

def FedAS(args, train_loader_list, test_loader_list):
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
    local_nodes = [None for i in range(args.num_users)]
    for round in tqdm(range(args.iters)):
        print(f'\n | Global Training Round : {round} |\n')
        # select clients
        client_weights = []
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                if round == 0:
                    local_nodes[idx2] = LocalUpdate(args=args)
                local_models[idx1 * len(datasets_name) + idx2], client_weight = local_nodes[idx2].update_weights_FedAS(model=copy.deepcopy(global_model), train_loader=train_loader_list[idx1 * len(datasets_name) + idx2])
                client_weights.append(client_weight)
        FIM_weight_list = [FIM_value/sum(client_weights) for FIM_value in client_weights]
        # update global model
        global_model, _ = communication(args, copy.deepcopy(global_model), copy.deepcopy(local_models), FIM_weight_list)
        loss_temp = [0 for i in range(len(datasets_name))]
        acc_temp = [0 for i in range(len(datasets_name))]
        for idx1 in range(args.num_users // len(datasets_name)):
            with torch.no_grad():
                for idx2 in range(len(datasets_name)):
                    local_test = LocalTest(args=args)
                    loss, _ = local_test.test_inference(args, copy.deepcopy(local_models[idx2]), train_loader_list[idx1 * len(datasets_name) + idx2])
                    loss_temp[idx2] += loss
                    _, acc = local_test.test_inference(args, copy.deepcopy(local_models[idx2]), test_loader_list[idx2])
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
        for _ in range(args.adcol_epoch):
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

def FedHEAL(args, train_loader_list, test_loader_list):
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
    L = torch.stack([torch.zeros(sum(p.numel() for p in global_model.parameters() if p.requires_grad)) for i in range(args.num_users)], dim = 0).to(args.device)
    for round in tqdm(range(args.iters)):
        diff_dict = []
        print(f'\n | Global Training Round : {round} |\n')
        # select clients
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                local_node = LocalUpdate(args=args)
                local_models[idx1 * len(datasets_name) + idx2] = local_node.update_weights(model=copy.deepcopy(global_model), train_loader=train_loader_list[idx1 * len(datasets_name) + idx2])
                # return difference of localmode and global model
                diff_dict_m = {}
                for (name_a, param_a), (name_b, param_b) in zip(local_models[idx1 * len(datasets_name) + idx2].named_parameters(), global_model.named_parameters()):
                    assert name_a == name_b
                    diff_dict_m[name_a] = param_a.data - param_b.data 
                diff_dict.append(diff_dict_m)
        diff_dict_flatten = torch.stack([torch.cat([p.view(-1) for p in diff_dict_m.values()], dim=0) for diff_dict_m in diff_dict], dim=0)
        if round > 0:
            q, L = calculate_q(args, L, diff_dict_flatten, round, client_weights)
        else:
            q = torch.tensor(client_weights).unsqueeze(dim=1).to(args.device) * torch.ones_like(diff_dict_flatten)
        # update global model
        if round != 0:
            global_model = communication_heal(args, copy.deepcopy(global_model), copy.deepcopy(diff_dict_flatten), q)
        else:
            global_model, _ = communication(args, copy.deepcopy(global_model), copy.deepcopy(local_models), client_weights)
        loss_temp = [0 for i in range(len(datasets_name))]
        acc_temp = [0 for i in range(len(datasets_name))]
        for idx1 in range(args.num_users // len(datasets_name)):
            with torch.no_grad():
                for idx2 in range(len(datasets_name)):
                    local_test = LocalTest(args=args)
                    loss, _ = local_test.test_inference(args, copy.deepcopy(local_models[idx2]), train_loader_list[idx1 * len(datasets_name) + idx2])
                    loss_temp[idx2] += loss
                    _, acc = local_test.test_inference(args, copy.deepcopy(local_models[idx2]), test_loader_list[idx2])
                    acc_temp[idx2] += acc
        for idx in range(len(datasets_name)):
            print('{:<11s} | train loss: {:.4f} | Test Acc: {:.4f}'.format(datasets_name[idx], loss_temp[idx] / (args.num_users // len(datasets_name)), acc_temp[idx] / (args.num_users // len(datasets_name))))
            train_loss[datasets_name[idx]].append(copy.deepcopy(loss_temp[idx] / (args.num_users // len(datasets_name))))
            accuracy_list[datasets_name[idx]].append(copy.deepcopy(acc_temp[idx] / (args.num_users // len(datasets_name))))
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
                local_models[idx1 * len(datasets_name) + idx2], features_label= local_node.update_weights_ours_debug(round, model=copy.deepcopy(local_models[idx1 * len(datasets_name) + idx2]), discriminator=copy.deepcopy(discriminator), classifier_model=copy.deepcopy(global_classifier), train_loader=train_loader_list[idx1 * len(datasets_name) + idx2],global_proto=global_proto, momentum = args.momentum)
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
        if sum(acc_temp) / len(acc_temp) > best_acc:
            for idx, dataset in enumerate(datasets_name):
                model_paths = f"weights/{args.seed}/{args.dataset}/"
                os.makedirs(model_paths, exist_ok=True)
                model_save_path = f"best_local_model_{dataset}.pth"
                torch.save(local_models[idx].state_dict(), model_paths + model_save_path)
            best_acc = sum(acc_temp) / len(acc_temp)
        # update global classifier layer
        features_label_dataset = data_utils.FeatureDataset(features_labels)
        features_label_loader = torch.utils.data.DataLoader(features_label_dataset, batch_size=args.batch, shuffle=True)  
        loss_func = nn.CrossEntropyLoss()
        global_classifier.train()
        for _ in range(args.adcol_epoch):
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
        for _ in range(args.adcol_epoch):
            for x, y in features_loader:
                x, y = x.to(args.device).float(), y.to(args.device).long()
                y_pred = discriminator(x)
                loss2 = loss_func(y_pred, y).mean()
                discriminator_optimizer.zero_grad()
                loss2.backward()
                discriminator_optimizer.step()
    if args.exp == 4:
        file_path = f'pkl/{args.dataset}/{args.mode}_{args.seed}_features_labels.pkl'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(features_labels, f)
    return train_loss, accuracy_list, datasets_name

def ablation1(args, train_loader_list, test_loader_list):
    # 进行不同损失函数组合，并保留原始特征-标签-客户端id三部分数据
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
                # 生成mask
                features = features_origin[idx1 * len(datasets_name) + idx2][0]
                labels = features_origin[idx1 * len(datasets_name) + idx2][1]
                size = features.shape[1]
                mask = generate_random_tensor(size, p=0.9).to(args.device)
                # 计算联邦梯度
                # 在0.8-1之间随机产生一个随机数
                lam = np.round(args.uniform_left + args.uniform_right * np.random.random(), 2)
                features_noise = []
                for idx, label in enumerate(labels):
                    features_noise.append(lam * features[idx] + (1 - lam) * global_proto[label.item()])
                features_noise = torch.stack(features_noise, dim=0)
                # 加入mask
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
        # 在准确度最高时保存原始特征
        if np.average(acc_temp) > best_acc:
            file_path = f'pkl/{args.dataset}/{args.mode}_{args.seed}_features_labels_idx_loss{args.loss_component}.pkl'
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # 获取训练结束后的特征-标签-客户端id
            # 将features_labels的数据转移到cpu上
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
        for _ in range(args.adcol_epoch):
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
            for _ in range(args.adcol_epoch):
                for x, y in features_loader:
                    x, y = x.to(args.device).float(), y.to(args.device).long()
                    y_pred = discriminator(x)
                    loss2 = loss_func(y_pred, y).mean()
                    discriminator_optimizer.zero_grad()
                    loss2.backward()
                    discriminator_optimizer.step()
    return train_loss, accuracy_list, datasets_name

def ablation2(args, train_loader_list, test_loader_list):
    # 比较替换和不替换本地分类器的结果
    # 只对全局分类器进行训练，计算CE+KL+infoNCE损失
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
                # 生成mask
                features = features_labels[idx1 * len(datasets_name) + idx2][0]
                labels = features_labels[idx1 * len(datasets_name) + idx2][1]
                size = features.shape[1]
                mask = generate_random_tensor(size, p=0.9).to(args.device)
                # 计算联邦梯度
                # 在0.8-1之间随机产生一个随机数
                lam = np.round(args.uniform_left + args.uniform_right * np.random.random(), 2)
                features_noise = []
                for idx, label in enumerate(labels):
                    features_noise.append(lam * features[idx] + (1 - lam) * global_proto[label.item()])
                features_noise = torch.stack(features_noise, dim=0)
                # 加入mask
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
            for _ in range(args.adcol_epoch):
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
        for _ in range(args.adcol_epoch):
            for x, y in features_loader:
                x, y = x.to(args.device).float(), y.to(args.device).long()
                y_pred = discriminator(x)
                loss2 = loss_func(y_pred, y).mean()
                discriminator_optimizer.zero_grad()
                loss2.backward()
                discriminator_optimizer.step()
    return train_loss, accuracy_list, datasets_name

def ablation3(args, train_loader_list, test_loader_list):
    # 消融实验3 安全隐私问题 不同噪声添加方式
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
        features_labels = []
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                local_node = LocalUpdate(args=args)
                local_models[idx1 * len(datasets_name) + idx2], features_label= local_node.update_weights_ours_debug(round, model=copy.deepcopy(local_models[idx1 * len(datasets_name) + idx2]), discriminator=copy.deepcopy(discriminator), classifier_model=copy.deepcopy(global_classifier), train_loader=train_loader_list[idx1 * len(datasets_name) + idx2],global_proto=global_proto, momentum = args.momentum)
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
        ## must
        global_proto = get_mean(args, local_protos, num)

        if args.mix_mode == 1:
            for idx1 in range(args.num_users // len(datasets_name)):
                for idx2 in range(len(datasets_name)):
                    features = features_labels[idx1 * len(datasets_name) + idx2][0]
                    labels = features_labels[idx1 * len(datasets_name) + idx2][1]
                    features_labels[idx1 * len(datasets_name) + idx2] = [features, labels]
                    ids = torch.ones(features.shape[0]) * (idx1 * len(datasets_name) + idx2)
                    features_idx.append([features, ids])
        elif args.mix_mode == 2:
            # 添加高斯噪声
            for idx1 in range(args.num_users // len(datasets_name)):
                for idx2 in range(len(datasets_name)):
                    features = features_labels[idx1 * len(datasets_name) + idx2][0]
                    labels = features_labels[idx1 * len(datasets_name) + idx2][1]
                    # 添加噪声
                    noise = torch.randn_like(features)
                    features_noise = features + noise
                    features_labels[idx1 * len(datasets_name) + idx2] = [features_noise, labels]
                    ids = torch.ones(features.shape[0]) * (idx1 * len(datasets_name) + idx2)
                    features_idx.append([features_noise, ids])
        elif args.mix_mode == 3:
            # 添加原型混合信息
            for idx1 in range(args.num_users // len(datasets_name)):
                for idx2 in range(len(datasets_name)):
                    # 生成mask
                    features = features_labels[idx1 * len(datasets_name) + idx2][0]
                    labels = features_labels[idx1 * len(datasets_name) + idx2][1]
                    size = features.shape[1]
                    mask = generate_random_tensor(size, p=0.9).to(args.device)
                    # 计算联邦梯度
                    # 在0.8-1之间随机产生一个随机数
                    lam = np.round(args.uniform_left + args.uniform_right * np.random.random(), 2)
                    features_noise = []
                    for idx, label in enumerate(labels):
                        features_noise.append(lam * features[idx] + (1 - lam) * global_proto[label.item()])
                    features_noise = torch.stack(features_noise, dim=0)
                    # 加入mask
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
        # 在准确度最高时保存混合特征
        if np.average(acc_temp) > best_acc:
            file_path = f'pkl/debug/{args.dataset}/{args.mode}_{args.seed}_features_labels_idx_NoiseType{args.mix_mode}.pkl'
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # 获取训练结束后的特征-标签-客户端id
            # 将features_labels的数据转移到cpu上
            for i in range(args.num_users):
                features_labels[i][0] = features_labels[i][0].to("cpu")
                features_labels[i][1] = features_labels[i][1].to("cpu")
            with open(file_path, 'wb') as f:
                pickle.dump(features_labels, f)
            for i in range(args.num_users):
                features_labels[i][0] = features_labels[i][0].to(args.device)
                features_labels[i][1] = features_labels[i][1].to(args.device)
            best_acc = np.average(acc_temp) 
        # update global classifier layer
        features_label_dataset = data_utils.FeatureDataset(features_labels)
        features_label_loader = torch.utils.data.DataLoader(features_label_dataset, batch_size=args.batch, shuffle=True)  
        loss_func = nn.CrossEntropyLoss()
        global_classifier.train()
        for _ in range(args.adcol_epoch):
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
        for _ in range(args.adcol_epoch):
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
    torch.backends.cudnn.deterministic = True  # 使用确定性算法
    torch.backends.cudnn.benchmark = False     # 关闭 cuDNN 自动优化选择算法
    args.device = args.device if torch.cuda.is_available() else 'cpu'

def fed_main(args):
    torch.cuda.set_device(args.device)
    # write log information
    if args.exp == 1:
        # 实验，比较不同算法在office/digit/PACS数据集上的准确度,三次实验求平均值和标准差，随机种子从0-2，客户端数量等同于数据集数量
        # 参数设置
        datasets = ["digit"]
        # modes = ["SingleSet", "fedavg", "fedprox", "perfedavg", "fedrep", "fedproto", "fedBN", "moon", "adcol", "RUCR", "ours_RUCR", "ours_RUCR_banlance", "ours", "ours_N",  "ours_RUCR_fc"]
        # important = ["SingleSet", "fedBN", "fedproto", "adcol", "ours_RUCR"]
        # modes = ["SingleSet", "fedavg"]
        # modes = ["fedproto", "fedprox"]
        # modes = ["fedBN", "moon"]
        # modes = ["ours_RUCR",  "RUCR"]
        modes = "ours"
        # modes = ["perfedavg", "fedrep"]
        # modes = ["ours_RUCR"]
        seeds = [0,1,2]
        args.iters = 100
        args.lr = 0.01
        for dataset in datasets:
            args.mode = modes
            args.dataset = dataset
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
                    args.batch = 32
                    args.wk_iters = 10
                    args.adcol_epoch = 3
                    args.adcol_mu = 0.1
                    args.adcol_beta = 0.1
                    datasets_name = ["amazon", "caltech", "dslr", "webcam"]
                    train_loader_list, test_loader_list = prepare_data_office_feature_noniid(args=args)
                    num_list = compute_num_list(args, train_loader_list)
                elif args.dataset == "digit":
                    args.num_classes = 10
                    args.num_users = 5
                    args.size = 28
                    args.batch = 64
                    args.wk_iters = 5
                    args.adcol_mu = 0.7
                    args.adcol_beta = 0.3
                    args.adcol_epoch = 1
                    datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
                    train_loader_list, test_loader_list = prepare_data_digit_feature_noniid(args=args)
                    num_list = compute_num_list(args, train_loader_list)
                elif args.dataset == "PACS":
                    args.num_classes = 7
                    args.num_users = 4
                    args.size = 64
                    args.batch = 32
                    args.wk_iters = 10
                    args.adcol_mu = 0.1
                    args.adcol_beta = 0.1
                    args.adcol_epoch = 3
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
                elif args.mode == "fedpcl":
                    train_loss, accuracy_list, datasets_name = fedpcl(args, train_loader_list, test_loader_list, num_list)
                elif args.mode == "fed_heal":
                    args.lr = 1e-3
                    if args.dataset == "digit":
                        args.beta = 0.4
                        args.tau = 0.3
                    elif args.dataset == "office":
                        args.beta = 0.4
                        args.tau = 0.4
                    elif args.dataset == "PACS":
                        args.beta = 0.4
                        args.tau = 0.4
                    excel_file = save_path + f"/test_acc_lr{args.lr}_tau{args.tau}_beta{args.beta}.xlsx"
                    if not os.path.exists(excel_file):
                        pd.DataFrame({"dataset": datasets_name}).to_excel(excel_file, index=False)
                    train_loss, accuracy_list, datasets_name = FedHEAL(args, train_loader_list, test_loader_list)
                elif args.mode == "ours":
                    train_loss, accuracy_list, datasets_name = ours(args, train_loader_list, test_loader_list)
                test_avg_acc = []
                df = pd.DataFrame(accuracy_list)
                # 计算每一行的平均值，并添加到最后一列
                df['average'] = df.mean(axis=1)
                # 保存 DataFrame 到 Excel 文件
                # path = f'{args.dataset}_{args.mode}_accuracy_results.xlsx'
                # df.to_excel(path, index=False)
                for idx in range(len(accuracy_list[datasets_name[0]])):
                    test_avg_acc.append(sum([accuracy_list[dataset][idx] for dataset in datasets_name]) / len(datasets_name)) 
                max_value = np.max(test_avg_acc)
                indices = np.where(np.array(test_avg_acc) == max_value)[0]
                best_step = indices[-1]
                for dataset_name in datasets_name:
                    data[f"seed:{seed}"].append(accuracy_list[dataset_name][best_step])
                with pd.ExcelWriter(excel_file, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                    existing_data = pd.read_excel(excel_file)
                    new_data = pd.DataFrame(data)
                    # 合并已有数据和新 `seed` 数据
                    combined_data = pd.concat([existing_data, new_data], axis=1)
                    combined_data.to_excel(writer, index=False)
    elif args.exp == 2:
        # 调参 比较不同mu值下的fedGac在office/PACS/digit数据集上的平均准确度
        # 参数设置
        args.dataset = "PACS"
        args.mode = "ours"
        seeds = [0,1,2]
        args.iters = 100
        args.lr = 0.01
        adcol_mus = [0.1,0.5,1]
        args.save_path = f"../result/exp{args.exp}/{args.dataset}/debug/"
        os.makedirs(args.save_path, exist_ok=True)
        for adcol_mu in adcol_mus:
            args.adcol_mu = adcol_mu
            for seed in seeds:
                data = collections.defaultdict(list)
                args.seed = seed
                set_seed(args)
                if args.dataset == "office":
                    args.num_classes = 10
                    args.num_users = 4
                    args.size = 64
                    args.batch = 32
                    args.wk_iters = 10
                    args.adcol_epoch = 3
                    args.adcol_mu = 0.1
                    args.adcol_beta = 0.1
                    datasets_name = ["amazon", "caltech", "dslr", "webcam"]
                    train_loader_list, test_loader_list = prepare_data_office_feature_noniid(args=args)
                    num_list = compute_num_list(args, train_loader_list)
                elif args.dataset == "digit":
                    args.num_classes = 10
                    args.num_users = 5
                    args.size = 28
                    args.batch = 64
                    args.wk_iters = 5
                    args.adcol_mu = 0.5
                    args.adcol_beta = 0.1
                    args.adcol_epoch = 1
                    datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
                    train_loader_list, test_loader_list = prepare_data_digit_feature_noniid(args=args)
                    num_list = compute_num_list(args, train_loader_list)
                elif args.dataset == "PACS":
                    args.num_classes = 7
                    args.num_users = 4
                    args.size = 64
                    args.batch = 32
                    args.wk_iters = 10
                    args.adcol_mu = 0.1
                    args.adcol_beta = 0.1
                    args.adcol_epoch = 3
                    datasets_name = ["art_painting", "cartoon", "photo", "sketch"]
                    train_loader_list, test_loader_list = prepare_data_PACS_feature_noniid(args=args)
                    num_list = compute_num_list(args, train_loader_list)
                print(args)
                save_path = args.save_path + f"{args.adcol_mu}"
                os.makedirs(save_path, exist_ok=True)
                excel_file = save_path + "/test_acc.xlsx"
                if not os.path.exists(excel_file):
                    pd.DataFrame({"dataset": datasets_name}).to_excel(excel_file, index=False)
                if args.mode == "ours":
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
                    # 合并已有数据和新 `seed` 数据
                    combined_data = pd.concat([existing_data, new_data], axis=1)
                    combined_data.to_excel(writer, index=False)
    elif args.exp == 3:
        # 消融实验，所有消融实验的随机种子都固定为0.
        # 消融实验1 比较不同损失函数的组合，并描述只有CE损失的特征分布， 以及不同组合下的特征分布，使用t-sne来描述(先保留特征-标签-客户端id)
        # 1) CE
        # 2) CE + infoNCE
        # 3) CE + KL
        # 4) CE + infoNCE + KL
        # 参数设置
        args.dataset = "office"
        args.mode = "ablation1"
        args.seed = 0
        args.iters = 100
        args.lr = 0.01
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
                args.batch = 32
                args.wk_iters = 10
                args.adcol_epoch = 3
                args.adcol_mu = 0.1
                args.adcol_beta = 0.1
                datasets_name = ["amazon", "caltech", "dslr", "webcam"]
                train_loader_list, test_loader_list = prepare_data_office_feature_noniid(args=args)
                num_list = compute_num_list(args, train_loader_list)
            elif args.dataset == "digit":
                args.num_classes = 10
                args.num_users = 5
                args.size = 28
                args.batch = 64
                args.wk_iters = 5
                args.adcol_mu = 0.7
                args.adcol_beta = 0.3
                args.adcol_epoch = 1
                datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
                train_loader_list, test_loader_list = prepare_data_digit_feature_noniid(args=args)
                num_list = compute_num_list(args, train_loader_list)
            elif args.dataset == "PACS":
                args.num_classes = 7
                args.num_users = 4
                args.size = 64
                args.batch = 32
                args.wk_iters = 10
                args.adcol_mu = 0.1
                args.adcol_beta = 0.1
                args.adcol_epoch = 3
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
                # 合并已有数据和新 `seed` 数据
                combined_data = pd.concat([existing_data, new_data], axis=1)
                combined_data.to_excel(writer, index=False)
    elif args.exp == 4:
        # 消融实验，所有消融实验的随机种子都固定为0.
        # 消融实验2 是否替换本地分类器
        # 1) 替换本地分类器
        # 2) 不替换本地分类器
        # 参数设置
        args.dataset = "office"
        args.mode = "ablation2"
        args.seed = 0
        args.iters = 100
        args.lr = 0.01
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
                args.batch = 32
                args.wk_iters = 10
                args.adcol_epoch = 3
                args.adcol_mu = 0.1
                args.adcol_beta = 0.1
                datasets_name = ["amazon", "caltech", "dslr", "webcam"]
                train_loader_list, test_loader_list = prepare_data_office_feature_noniid(args=args)
                num_list = compute_num_list(args, train_loader_list)
            elif args.dataset == "digit":
                args.num_classes = 10
                args.num_users = 5
                args.size = 28
                args.batch = 64
                args.wk_iters = 5
                args.adcol_mu = 0.7
                args.adcol_beta = 0.3
                args.adcol_epoch = 1
                datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
                train_loader_list, test_loader_list = prepare_data_digit_feature_noniid(args=args)
                num_list = compute_num_list(args, train_loader_list)
            elif args.dataset == "PACS":
                args.num_classes = 7
                args.num_users = 4
                args.size = 64
                args.batch = 32
                args.wk_iters = 10
                args.adcol_mu = 0.1
                args.adcol_beta = 0.1
                args.adcol_epoch = 3
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
                # 合并已有数据和新 `seed` 数据
                combined_data = pd.concat([existing_data, new_data], axis=1)
                combined_data.to_excel(writer, index=False)
    elif args.exp == 5:
        # 消融实验，所有消融实验的随机种子都固定为0.
        # 消融实验3 安全隐私问题
        # 1) 不添加任何隐私保护方式，使用原始特征
        # 2) 添加高斯噪声
        # 3) 添加原型混合信息
        args.dataset = "office"
        args.mode = "ablation3"
        args.seed = 0
        args.iters = 100
        args.lr = 0.01
        mix_modes = [1, 2, 3]
        args.save_path = f"../result/exp{args.exp}/{args.dataset}/"
        os.makedirs(args.save_path, exist_ok=True)
        for mix_mode in mix_modes:
            args.mix_mode = mix_mode
            data = collections.defaultdict(list)
            set_seed(args)
            if args.dataset == "office":
                args.num_classes = 10
                args.num_users = 4
                args.size = 64
                args.batch = 32
                args.wk_iters = 10
                args.adcol_epoch = 3
                args.adcol_mu = 0.1
                args.adcol_beta = 0.1
                datasets_name = ["amazon", "caltech", "dslr", "webcam"]
                train_loader_list, test_loader_list = prepare_data_office_feature_noniid(args=args)
                num_list = compute_num_list(args, train_loader_list)
            elif args.dataset == "digit":
                args.num_classes = 10
                args.num_users = 5
                args.size = 28
                args.batch = 64
                args.wk_iters = 5
                args.adcol_mu = 0.7
                args.adcol_beta = 0.3
                args.adcol_epoch = 1
                datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
                train_loader_list, test_loader_list = prepare_data_digit_feature_noniid(args=args)
                num_list = compute_num_list(args, train_loader_list)
            elif args.dataset == "PACS":
                args.num_classes = 7
                args.num_users = 4
                args.size = 64
                args.batch = 32
                args.wk_iters = 10
                args.adcol_mu = 0.1
                args.adcol_beta = 0.1
                args.adcol_epoch = 3
                datasets_name = ["art_painting", "cartoon", "photo", "sketch"]
                train_loader_list, test_loader_list = prepare_data_PACS_feature_noniid(args=args)
                num_list = compute_num_list(args, train_loader_list)
            print(args)
            save_path = args.save_path + f"noisetype_{args.mix_mode}"
            os.makedirs(save_path, exist_ok=True)
            excel_file = save_path + "/test_acc.xlsx"
            if not os.path.exists(excel_file):
                pd.DataFrame({"dataset": datasets_name}).to_excel(excel_file, index=False)
            if args.mode == "ablation3":
                train_loss, accuracy_list, datasets_name = ablation3(args, train_loader_list, test_loader_list)
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
                # 合并已有数据和新 `seed` 数据
                combined_data = pd.concat([existing_data, new_data], axis=1)
                combined_data.to_excel(writer, index=False)
    elif args.exp == 6:
        args.dataset = "PACS"
        modes = ["ours"]
        seeds = [0,1,2]
        args.iters = 100
        args.lr = 0.01
        args.save_path = f"../result/exp{args.exp}/{args.dataset}/format/{args.mode}/"
        os.makedirs(args.save_path, exist_ok=True)
        for mode in modes:
            args.mode = mode
            for seed in seeds:
                data = collections.defaultdict(list)
                args.seed = seed
                set_seed(args)
                if args.dataset == "office":
                    args.num_classes = 10
                    args.num_users = 4
                    args.size = 64
                    args.batch = 32
                    args.wk_iters = 10
                    args.adcol_epoch = 3
                    args.adcol_mu = 0.1
                    args.adcol_beta = 0.1
                    datasets_name = ["amazon", "caltech", "dslr", "webcam"]
                    train_loader_list, test_loader_list = prepare_data_office_feature_noniid(args=args)
                    num_list = compute_num_list(args, train_loader_list)
                elif args.dataset == "digit":
                    args.num_classes = 10
                    args.num_users = 5
                    args.size = 28
                    args.batch = 64
                    args.wk_iters = 5
                    args.adcol_mu = 0.7
                    args.adcol_beta = 0.3
                    args.adcol_epoch = 1
                    datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
                    train_loader_list, test_loader_list = prepare_data_digit_feature_noniid(args=args)
                    num_list = compute_num_list(args, train_loader_list)
                elif args.dataset == "PACS":
                    args.num_classes = 7
                    args.num_users = 4
                    args.size = 64
                    args.batch = 32
                    args.wk_iters = 10
                    args.adcol_mu = 0.1
                    args.adcol_beta = 0.1
                    args.adcol_epoch = 3
                    datasets_name = ["art_painting", "cartoon", "photo", "sketch"]
                    train_loader_list, test_loader_list = prepare_data_PACS_feature_noniid(args=args)
                    num_list = compute_num_list(args, train_loader_list)
                print(args)
                save_path = args.save_path + f"{args.lr}_{args.adcol_mu}_{args.adcol_beta}"
                os.makedirs(save_path, exist_ok=True)
                excel_file = save_path + "/test_acc.xlsx"
                if not os.path.exists(excel_file):
                    pd.DataFrame({"dataset": datasets_name}).to_excel(excel_file, index=False)
                if args.mode == "ours":
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
                    # 合并已有数据和新 `seed` 数据
                    combined_data = pd.concat([existing_data, new_data], axis=1)
                    combined_data.to_excel(writer, index=False)

if __name__ == "__main__":
    args = args_parser()
    args.seed = 1
    fed_main(args)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    