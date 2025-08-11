from utils.load_data import *
from utils.eval import *
from utils.tools import *
from network import *
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import random
import torch.nn.functional as F
import utils.log
import os
from datetime import datetime
import torch.nn as nn

torch.multiprocessing.set_sharing_strategy('file_system')

# 64:  17~32  96:29~56  128: 41~80  160:55~108  192:69~136  224:82~162  256: 95~188 
# margin: 64:30  96:45  128:60   160:75  192: 90  224:110   256:125
def get_config():
    config = {
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "batch_size": 64,
        "epoch": 10,
        "device": torch.device("cuda:0"),
        "bit_list": [64, 128, 256],
        "n_class": 56441, # 56441 categorys in trainset
        "1m": 0,
        # private params
        "alpha": 0.01,
        "margin_list": [30, 60, 125],
    }
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
        
        target_code = self.label2center(y).to(config["device"])
        target_code = self.sign_with_random_zeros(target_code).to(config["device"])

        distance_squared = (h - target_code.detach()).pow(2)
        euclidean_distance_sum = distance_squared.sum(dim=1)

        center_loss = euclidean_distance_sum.sum() / 2
        # quantization loss
        # center_loss = (h.abs() - 1).pow(2).mean()
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
        hash_center = self.codebook[y.argmax(axis=1)].to(config["device"])
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

        # print(dist_q)
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


def train4gldv2(config, bit, margin, use_1m):
    set_seed()
    device = config["device"]
    train_loader, num_train = get_train_data(config)
    config["num_train"] = num_train
    hidden_dim = 2048 + bit

    hashnet = HASHMLP(2048,
                      [hidden_dim],
                      bit,
                      use_BN=True, use_tanh=True).to(device)
    optimizer1 = config["optimizer"]["type"](hashnet.parameters(), **(config["optimizer"]["optim_params"]))


    Lp = PairLoss(config, bit, margin)
    Lcen = CenterLoss(config, bit)

    for epoch in range(config["epoch"]):
        hashnet.train()
        pair_loss = 0
        center_loss = 0

        for image, label, ind in tqdm(train_loader):
            features = image.squeeze(1).to(device)
            label = label.to(device)
            label = F.one_hot(label, num_classes=config["n_class"]).float().to(device)
            optimizer1.zero_grad()
            code = hashnet(features)

            loss1 = Lp(code, label)
            loss2 = Lcen(code, label, config["alpha"],)
            loss = loss1 + loss2

            loss.backward()
            optimizer1.step()

            pair_loss += loss1.item()
            center_loss += loss2.item()

        print('[%3d] alpha: %.2f, pair: %.4f, center: %.4f' % (
            epoch, config["alpha"], pair_loss / len(train_loader), center_loss / len(train_loader)))
        if (epoch + 1) % 2 == 0:
            data_dir = "./datasets"
            dataset_list = ["roxford5k", "rparis6k"]
            scale_list = [0.7071, 1.0, 1.4142]
            test_model(hashnet, data_dir, dataset_list, scale_list, device, use_1m)

if __name__ == "__main__":
    config = get_config()
    only_show_margin = False

    if only_show_margin:
        for bit in config["bit_list"]:
            show_margin(bit, config)
        exit(0)
        # 64:  17~32  96:29~56  128: 41~80  160:55~108  192:69~136  224:82~162  256: 95~188 
    for i in range(len(config["bit_list"])):
        train4gldv2(config, config["bit_list"][i], config["margin_list"][i], config["1m"])
