from utils.tools import *
from network import *
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import random
import torch.nn.functional as F
import os
from datetime import datetime
import torch.nn as nn
import math

torch.multiprocessing.set_sharing_strategy('file_system')

# 4~6       10~18      24~46 
# 16:6    32:13   64: 29
def get_config():
    config = {
        "alpha": 0.01,
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "info": "[Hamming Ball Hashing]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
        "net": ResNet,
        "dataset": "imagenet",
        "epoch": 100,
        "test_map": 20,
        "device": torch.device("cuda:1"),
        "bit_list": [16, 32, 64],
        "margin_list": [6, 13, 29],
    }
    config = config_dataset(config)
    return config

def set_seed():
    seed_value = 42
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CenterLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(CenterLoss, self).__init__()
        self.codebook = self.init_codebook(config["n_class"], bit).to(config["device"]).requires_grad_(False)
        self.bit = bit
        self.config = config

    def forward(self, h, y, alpha):
        for i in range(h.shape[0]):
            h_cloned = h[i].clone().detach().requires_grad_(False)
            self.codebook[y[i].argmax(axis=0).item()] += h_cloned.sign()
        
        target_code = self.label2center(y).to(self.config["device"])
        target_code = self.sign_with_random_zeros(target_code).to(self.config["device"])

        distance_squared = (h - target_code.detach()).pow(2)
        euclidean_distance_sum = distance_squared.sum(dim=1)

        center_loss = euclidean_distance_sum.sum() / 2
        return center_loss * alpha

    def sign_with_random_zeros(self, t):
        return torch.where(
            t > 0,
            torch.ones_like(t),
            torch.where(t < 0,
                        -torch.ones_like(t),
                        torch.randint(0, 2, t.size(), device=t.device) * 2 - 1)
        )

    def label2center(self, y):
        hash_center = self.codebook[y.argmax(axis=1)].to(self.config["device"])
        return hash_center

    def init_codebook(self, n_class, bit):
        prob = torch.ones(n_class, bit) * 0.5
        codebook = torch.bernoulli(prob) * 2. - 1.
        codebook = codebook.sign()
        return codebook

class PairLoss(torch.nn.Module):
    def __init__(self, config, bit, margin):
        super(PairLoss, self).__init__()
        self.config = config
        self.bit = bit
        self.margin = margin

    def forward(self, h, y):
        margin = self.margin

        S = (y @ y.t() > 0).float().to(self.config['device'])

        inner_product = torch.matmul(h, h.t()).to(self.config['device'])
        dist_q = 0.5 * (self.bit - inner_product)

        neg_mask = (1 - S) * (dist_q < margin).float()
        loss_neg = neg_mask * (margin - dist_q)
        loss_neg = loss_neg.sum()

        pos_mask = S.float()
        loss_pos = pos_mask * dist_q
        loss_pos = loss_pos.sum()
        loss = loss_neg + loss_pos

        return loss.mean() / self.bit

def combination_formula(n, k):
    if k == 0 or k == n:
        return 1
    if k > n:
        return 0
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

def get_sum_with_d(q, x):
    sum = 0
    for i in range(x):
        sum += combination_formula(q, i)
    return sum

def get_sum_with_e(q, x):
    sum = 0
    x = int((x - 1) / 2)
    for i in range(x + 1):
        sum += combination_formula(q, i)
    return sum

def show_margin(bit, config):
    ret = []
    for i in range(1, bit):
        if (2 ** bit / get_sum_with_d(bit, i) <= config["n_class"] <= 2 ** bit / get_sum_with_e(bit, i)):
            ret.append(i)
    print(f"bit={bit}, m={ret}")

def train(config, bit, margin):
    set_seed()
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data4imagenet(config)
    net = config["net"](bit).to(device)

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    Lp = PairLoss(config, bit, margin)
    Lcen = CenterLoss(config, bit)

    Best_mAP = 0

    for epoch in range(1, config["epoch"] + 1):
        pair_loss = 0
        center_loss = 0
        for image, label, ind in tqdm(train_loader):
            net.train()
            image = image.to(device)
            label = label.to(device).float()
            optimizer.zero_grad()
            code = net(image)
            
            loss1 = Lp(code, label)
            loss2 = Lcen(code, label, config["alpha"])
            loss = loss1 + loss2
            
            pair_loss += loss1.item()
            center_loss += loss2.item()
            
            loss.backward()
            optimizer.step()
        
        print('[%3d] alpha: %.2f, pair: %.4f, center: %.4f' % (
            epoch, config["alpha"], pair_loss / len(train_loader), center_loss / len(train_loader)))
        
        if (epoch > 10 and epoch % config["test_map"] == 0):
            Best_mAP = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset)

if __name__ == "__main__":
    config = get_config()
    print(config)
    only_show_margin = False

    if only_show_margin:
        for bit in config["bit_list"]:
            show_margin(bit, config)
        exit(0)
    
    for i in range(len(config["bit_list"])):
        train(config, config["bit_list"][i], config["margin_list"][i])