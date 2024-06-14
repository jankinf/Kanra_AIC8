#!/usr/bin/env python

from typing import Iterable, List, Optional, Callable
from PIL import Image
import torch
import torchvision.transforms as T
import argparse
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import torch.utils.data as data
from torch import nn
import os
import numpy as np
from robustbench.utils import load_model

ROOT = "./data"
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

CORRUPTIONS = ["brightness","contrast","defocus_blur","elastic_transform","fog","frost","gaussian_noise","glass_blur","impulse_noise","jpeg_compression","motion_blur","pixelate","shot_noise","snow","zoom_blur"]

##############################
#        PARAMETERS          #
##############################

BATCH_SIZE = 256
severity = 2
s0_num = 18000 # clean data
s2_num = 12000 # corruption by given severity
adv_num = 20000 # adversarial examples

##############################
#      END PARAMETERS        #
##############################


class CIFAR10C(Dataset):
    filename = "CIFAR-10-C"

    def __init__(self, root: str = ROOT, transform: Optional[Callable] = None, corruption_type: str = 'snow'):
        dataPath = os.path.join(root, '{}.npy'.format(corruption_type))
        labelPath = os.path.join(root, 'labels.npy')

        self.data = np.load(dataPath)[50000*(severity-1) : 50000*severity]
        self.label = np.load(labelPath).astype(np.long)
        self.transform = transform

    def __getitem__(self, idx):

        img = self.data[idx]
        img = Image.fromarray(img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        label = self.label[idx]

        return img, label

    def __len__(self):
        return self.data.shape[0]


def load_models(model_names):
    models = []
    for model_name, t_type in model_names:
        print(model_name)
        model = load_model(model_name=model_name, model_dir='./ckpt', dataset='cifar10', threat_model=t_type)
        models.append(model.cuda())
    return models


class ADV(Dataset):
    adv_x = "adv.npy"
    adv_y = "label.npy"
    adv_c_x = "adv_c.npy"
    adv_c_y = "label_c.npy"

    def __init__(self, data_path: str, transform: Optional[Callable] = None, with_c: bool = False):
        if with_c:
            dataPath = os.path.join(data_path, self.adv_c_x)
            labelPath = os.path.join(data_path, self.adv_c_y)
        else:
            dataPath = os.path.join(data_path, self.adv_x)
            labelPath = os.path.join(data_path, self.adv_y)

        self.data = np.load(dataPath)
        self.label = np.load(labelPath).astype(np.long)
        self.transform = transform

    def __getitem__(self, idx):

        img = self.data[idx] * 255
        img = Image.fromarray(img.astype(np.uint8)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        label = self.label[idx]

        return img, label

    def __len__(self):
        return self.data.shape[0]


def load_cifar() -> Iterable:
    dataset = datasets.CIFAR10(root='./data', transform=T.ToTensor(), train=True, download=True)
    testloader = data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE)
    return testloader

def load_cifar_c(corruption: str) -> Iterable:
    dataset = CIFAR10C(corruption_type=corruption, transform=T.ToTensor())
    testloader = data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE)
    return testloader

def load_adv(data_path: str) -> Iterable:
    dataset = ADV(data_path, transform=T.ToTensor())
    testloader = data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE)
    return testloader

def entropy(px):
    return -torch.sum(F.softmax(px, dim=1) * F.log_softmax(px, dim=1), dim=1)

def merge_pc(a, etp_idx, num=50000):
    res = a

    etp = res[etp_idx]
    idxs = torch.argsort(etp.view(-1), descending=True)[:num]

    for i in range(len(res)):
        res[i] = res[i][idxs]
    return res


def merge_data(a, offset=0, num=50000):
    res = a

    etp = res[2]
    idxs = torch.argsort(etp.view(-1), descending=True)[offset:offset+num]

    for i in range(len(res)):
        res[i] = res[i][idxs]
    return res

def load_adv(data_path: str) -> Iterable:
    dataset = ADV(data_path, transform=T.ToTensor())
    testloader = data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE)
    return testloader

def main_entropy_pc(model, cname, device=DEVICE):
    mdict = {}
    for i in range(10):
        mdict[i] = {"xs": [], "ys_s": [], "ys_h": [], "etp": [], "c": []}

    loader = load_cifar()
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            logits = model(inputs)
        softlabel = torch.nn.functional.softmax(logits, dim=1)
        etps = entropy(softlabel)
        for l in map(int, list(set(labels.detach().cpu().numpy()))):
            idx = (labels == l)
            mdict[l]['xs'].append(inputs[idx])
            mdict[l]['ys_s'].append(softlabel[idx])
            mdict[l]['ys_h'].append(labels[idx])
            mdict[l]['etp'].append(etps[idx])
            mdict[l]['c'].append(torch.ones_like(labels[idx]).to(labels.device))
    for l in range(10):
        mdict[l]['xs'] = torch.cat(mdict[l]['xs'], dim=0)
        mdict[l]['ys_s'] = torch.cat(mdict[l]['ys_s'], dim=0)
        mdict[l]['ys_h'] = torch.cat(mdict[l]['ys_h'], dim=0)
        mdict[l]['etp'] = torch.cat(mdict[l]['etp'], dim=0)
        mdict[l]['c'] = torch.cat(mdict[l]['c'], dim=0)

    for l in range(10):
        mdict[l]['xs'], mdict[l]['ys_s'], mdict[l]['ys_h'], mdict[l]['etp'], mdict[l]['c'] = merge_pc(
            [mdict[l]['xs'], mdict[l]['ys_s'], mdict[l]['ys_h'], mdict[l]['etp'], mdict[l]['c']],
            etp_idx=3,
            num=5000)

    offset = 0
    num = int(s0_num/10)
    xs1 = torch.cat([mdict[l]['xs'][offset:offset+num] for l in range(10)])
    ys_s1 = torch.cat([mdict[l]['ys_s'][offset:offset+num] for l in range(10)])
    ys_h1 = torch.cat([mdict[l]['ys_h'][offset:offset+num] for l in range(10)])
    etp1 = torch.cat([mdict[l]['etp'][offset:offset+num] for l in range(10)])
    c1 = torch.cat([mdict[l]['c'][offset:offset+num] for l in range(10)])

    for l in range(10):
        assert (ys_h1 == l).sum() == num

    print(c1.sum(), c1.numel())
    print(etp1.min(), etp1.max())
    l_1 = torch.argmin(etp1)
    l_2 = torch.argmax(etp1)
    print("least chaos: ", ys_s1[l_1], ys_s1[l_1].min(), ys_s1[l_1].max())
    print("most chaos: ", ys_s1[l_2], ys_s1[l_2].min(), ys_s1[l_2].max())
    xs = xs1.permute(0, 2, 3, 1).detach().cpu().numpy()
    ys = ys_h1.detach().cpu().numpy()
    xs = (xs * 255).astype(np.uint8)
    ys = np.eye(10)[ys]
    ys = ys.astype(np.float64)
    print(xs.shape, xs.dtype, ys.shape, ys.dtype)
    return xs, ys

def main_entropy_cor(model, cname, device=DEVICE):
    num_model = len(CORRUPTIONS)
    all_x, all_ys_s, all_ys_h, all_etp = [], [], [], []

    for corruption in CORRUPTIONS:
        testloader = load_cifar_c(corruption)
        print(corruption)
        xs, ys_s, ys_h, etp = [], [], [], []
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                logits = model(inputs)
            xs.append(inputs)
            ys_s.append(logits)
            ys_h.append(labels)
            etp.append(entropy(torch.nn.functional.softmax(logits, dim=1)))

        xs_cat = torch.cat(xs, dim=0)
        ys_s_cat = torch.cat(ys_s, dim=0)
        ys_h_cat = torch.cat(ys_h, dim=0)
        etp_cat = torch.cat(etp, dim=0)

        all_x.append(xs_cat)
        all_ys_s.append(ys_s_cat)
        all_ys_h.append(ys_h_cat)
        all_etp.append(etp_cat)

    all_x_cat = torch.cat(all_x, dim=0).reshape(num_model, 50000, 3, 32, 32)
    all_ys_s_cat = torch.cat(all_ys_s, dim=0).reshape(num_model, 50000, 10)
    all_ys_h_cat = torch.cat(all_ys_h, dim=0).reshape(num_model, 50000)
    all_etp_cat = torch.cat(all_etp, dim=0).reshape(num_model, 50000)

    print("mean entropy")
    print(all_etp_cat.mean(0))
    print("min entropy")
    print(all_etp_cat.min(0)[0])
    print("max entropy")
    print(all_etp_cat.max(0)[0])

    max_etp_idx = torch.argmax(all_etp_cat, dim=0)
    for i in range(num_model):
        num_samples = (max_etp_idx == i).sum().item()
        print("{} samples from {}".format(num_samples, CORRUPTIONS[i]))

    fx = torch.gather(all_x_cat, dim=0, index=max_etp_idx[None, :, None, None, None].expand(1, 50000, 3, 32, 32)).squeeze()
    fyh = torch.gather(all_ys_h_cat, dim=0, index=max_etp_idx[None, :].expand(1, 50000)).squeeze()

    etp = torch.gather(all_etp_cat, dim=0, index=max_etp_idx[None, :].expand(1, 50000)).squeeze()

    fx1, fyh1, _ = merge_data([fx, fyh, etp], offset = 50000-s2_num, num = s2_num)
    xs = fx1.permute(0, 2, 3, 1).detach().cpu().numpy()
    ys = fyh1.detach().cpu().numpy()
    xs = (xs * 255).astype(np.uint8)
    ys = np.eye(10)[ys]
    ys = ys.astype(np.float64)
    print(xs.shape, xs.dtype, ys.shape, ys.dtype)
    return xs, ys


def main_entropy_adv(model, cname, device=DEVICE):
    model_names = ['Diffenderfer2021Winning_LRR']
    num_model = len(model_names)
    all_x, all_ys_s, all_ys_h, all_etp = [], [], [], []
    for model_name in model_names:
        testloader = load_adv(f"./data/adv/{model_name}/AutoAttack")
        print(model_name)
        xs, ys_s, ys_h, etp = [], [], [], []
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                logits = model(inputs)
            xs.append(inputs)
            ys_s.append(logits)
            ys_h.append(labels)
            etp.append(entropy(torch.nn.functional.softmax(logits, dim=1)))

        xs_cat = torch.cat(xs, dim=0)
        ys_s_cat = torch.cat(ys_s, dim=0)
        ys_h_cat = torch.cat(ys_h, dim=0)
        etp_cat = torch.cat(etp, dim=0)

        all_x.append(xs_cat)
        all_ys_s.append(ys_s_cat)
        all_ys_h.append(ys_h_cat)
        all_etp.append(etp_cat)
    
    all_x_cat = torch.cat(all_x, dim=0).reshape(num_model, 50000, 3, 32, 32)
    all_ys_s_cat = torch.cat(all_ys_s, dim=0).reshape(num_model, 50000, 10)
    all_ys_h_cat = torch.cat(all_ys_h, dim=0).reshape(num_model, 50000)
    all_etp_cat = torch.cat(all_etp, dim=0).reshape(num_model, 50000)
    
    print("mean entropy")
    print(all_etp_cat.mean(0))
    print("min entropy")
    print(all_etp_cat.min(0)[0])
    print("max entropy")
    print(all_etp_cat.max(0)[0])
    
    max_etp_idx = torch.argmax(all_etp_cat, dim=0)
    for i in range(num_model):
        num_samples = (max_etp_idx == i).sum().item()
        print("{} samples from {}".format(num_samples, i))

    fx = torch.gather(all_x_cat, dim=0, index=max_etp_idx[None, :, None, None, None].expand(1, 50000, 3, 32, 32)).squeeze()
    fyh = torch.gather(all_ys_h_cat, dim=0, index=max_etp_idx[None, :].expand(1, 50000)).squeeze()
    
    etp = torch.gather(all_etp_cat, dim=0, index=max_etp_idx[None, :].expand(1, 50000)).squeeze()
    
    fx1, fyh1, _ = merge_data([fx, fyh, etp], offset = 50000-adv_num, num = adv_num)
    xs = fx1.permute(0, 2, 3, 1).detach().cpu().numpy()
    ys = fyh1.detach().cpu().numpy()
    xs = (xs * 255).astype(np.uint8)
    ys = np.eye(10)[ys]
    ys = ys.astype(np.float64)
    print(xs.shape, xs.dtype, ys.shape, ys.dtype)
    return xs, ys

if __name__ == "__main__":
    model_names = [('Diffenderfer2021Winning_LRR', 'corruptions')]
    assert len(model_names) == 1 and model_names[0][0] == 'Diffenderfer2021Winning_LRR'

    models = load_models(model_names)

    x_merged = []
    y_merged = []

    for model, (model_name, t_type) in zip(models, model_names):
        # Clean Data
        print("Selecting clean data...")
        x_clean, y_clean = main_entropy_pc(model, cname=f"{model_name}_{t_type}")
        if len(x_merged):
            x_merged = np.append(x_merged, x_clean, axis=0)
            y_merged = np.append(y_merged, y_clean, axis=0)
        else:
            x_merged, y_merged = x_clean, y_clean

        # Corruptions
        print("Selecting corruptions...")
        x_cor, y_cor = main_entropy_cor(model, cname=f"{model_name}_{t_type}")
        x_merged = np.append(x_merged, x_cor, axis=0)
        y_merged = np.append(y_merged, y_cor, axis=0)
        
        # Adversarial Examples
        print("Selecting adversarial examples...")
        x_adv, y_adv = main_entropy_adv(model, cname=f"{model_name}_{t_type}")
        x_merged = np.append(x_merged, x_adv, axis=0)
        y_merged = np.append(y_merged, y_adv, axis=0)

    np.save('./data.npy', x_merged)
    np.save('./label.npy', y_merged)
    print("saved!")

