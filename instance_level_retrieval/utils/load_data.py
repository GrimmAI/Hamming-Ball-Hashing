import numpy as np
import torch.utils.data as util_data
import torchvision.datasets as dsets
import torch


class gldv2EmbeddingList(object):
    def __init__(self, file_path, label_path):
        self.features = np.load(file_path)
        
        with open(label_path, 'r', encoding='utf-8') as file:  
            lines = file.readlines()  
        lines = [int(line.strip()) for line in lines]
        self.gt = lines
        
        self.len = len(self.features)

    def __getitem__(self, index):
        feature = self.features[index]
        gt = self.gt[index]
        return feature, gt, index

    def __len__(self):
        return self.len


def get_train_data(config):
    file_path = "./datasets/gldv2_clean/cvnet_train_gldv2clean.npy"
    label_path = "./datasets/gldv2_clean/gldv2_label.txt"
    dsets = {}
    dset_loaders = {}
    for data_set in ["train_set"]:
        dsets[data_set] = gldv2EmbeddingList(file_path, label_path)
        print(data_set, len(dsets[data_set]))
        dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                      batch_size=config["batch_size"],
                                                      shuffle= (data_set == "train_set") , num_workers=4, drop_last=True)

    return dset_loaders["train_set"], len(dsets["train_set"])
