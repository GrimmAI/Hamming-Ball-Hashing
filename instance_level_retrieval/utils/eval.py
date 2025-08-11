import numpy as np
import torch
from tqdm import tqdm
import torchvision.datasets as dsets
from utils.tools import *
from network import *
from utils.config_gnd import config_gnd

device = torch.device("cuda:0")


@torch.no_grad()
def test_revisitop(cfg, ks, ranks):
    # revisited evaluation
    gnd = cfg['gnd']
    ranks_E, ranks_M, ranks_H = ranks

    # evaluate ranks
    # search for easy
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['easy']])
        g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['hard']])
        gnd_t.append(g)
    mapE, apsE, mprE, prsE = compute_map(ranks_E, gnd_t, ks)

    # search for easy & hard
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
        g['junk'] = np.concatenate([gnd[i]['junk']])
        gnd_t.append(g)
    mapM, apsM, mprM, prsM = compute_map(ranks_M, gnd_t, ks)

    # search for hard
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['hard']])
        g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
        gnd_t.append(g)
    mapH, apsH, mprH, prsH = compute_map(ranks_H, gnd_t, ks)

    return (mapE, apsE, mprE, prsE), (mapM, apsM, mprM, prsM), (mapH, apsH, mprH, prsH)


@torch.no_grad()
def test_model(model, data_dir, dataset_list, scale_list, device, use_1m=0, print_fun=print):
    # print_fun is set for designating logging or print to save the results, please use no other para in PRINT.
    torch.backends.cudnn.benchmark = False
    model.eval()
    for dataset in dataset_list:
        text = '>> {}: Global Retrieval for scale {} with CVNet-Global'.format(dataset, str(scale_list))
        print_fun(text)
        if dataset == 'roxford5k':
            gnd_fn = 'gnd_roxford5k.pkl'
        elif dataset == 'rparis6k':
            gnd_fn = 'gnd_rparis6k.pkl'
        else:
            assert dataset

        if (dataset == "rparis6k"):
            rB = extract_feature4hash(model, "./datasets/rparis6k/rparis6k_db.npy", is_use_1m=use_1m, device=device)
            qB = extract_feature4hash(model, "./datasets/rparis6k/r50_rparis6k_query.npy", is_use_1m=use_1m, device=device)
        else:
            rB = extract_feature4hash(model, "./datasets/roxford5k/roxford5k_db.npy", is_use_1m=use_1m, device=device)
            qB = extract_feature4hash(model, "./datasets/roxford5k/r50_roxford5k_query.npy", is_use_1m=use_1m, device=device)


        cfg = config_gnd(dataset, data_dir)

        # perform search
        print_fun("perform global retrieval")
        # sim = np.dot(X, Q.T)
        # ranks = np.argsort(-sim, axis=0)

        hamm = CalcHammingDist4gldv2(rB, qB)
        ranks = np.argsort(hamm, axis=0)

        # revisited evaluation
        gnd = cfg['gnd']
        ks = [1, 5, 10]
        (mapE, apsE, mprE, prsE), (mapM, apsM, mprM, prsM), (mapH, apsH, mprH, prsH) = test_revisitop(cfg, ks,
                                                                                                      [ranks, ranks,
                                                                                                       ranks])

        print_fun('Global retrieval results: mAP E: {}, M: {}, H: {}'.format(np.around(mapE * 100, decimals=2),
                                                                             np.around(mapM * 100, decimals=2),
                                                                             np.around(mapH * 100, decimals=2)))


def extract_feature4hash(hashnet, path, is_use_1m, device):
    hashnet.eval()
    if is_use_1m == 0:
        img_feats_agg = np.load(path)
        img_feats_agg = torch.from_numpy(img_feats_agg)
    else:
        img_feats_agg1 = np.load(path)
        img_feats_agg2 = np.load("./datasets/1mdb.npy")
        img_feats_agg = np.concatenate((img_feats_agg1, img_feats_agg2), axis=0)
        img_feats_agg = torch.from_numpy(img_feats_agg)

    ans = []
    for item in img_feats_agg:
        hashcode = hashnet(item.unsqueeze(0).to(device))
        ans.append(hashcode.squeeze(0).data.sign().cpu())
    ans = [tensor.numpy() for tensor in ans]
    ans = np.vstack(ans)
    return ans


def compute_result4gldv2(hashnet, device, path):
    hashnet.eval()
    img_feats_agg = np.load(path)
    img_feats_agg = torch.from_numpy(img_feats_agg)
    ans = []
    for item in img_feats_agg:
        hashcode = hashnet(item.unsqueeze(0).to(device))
        ans.append(hashcode.squeeze(0).data.sign().cpu())
    ans = [tensor.numpy() for tensor in ans]
    ans = np.vstack(ans)
    return ans


# sim = np.dot(X.T, Q)
def CalcHammingDist4gldv2(X, Q):
    q = Q.shape[1]
    print(q)
    distH = 0.5 * (q - np.dot(X, Q.T))
    return distH
