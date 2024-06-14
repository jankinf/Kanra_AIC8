from typing import List
from robustbench.data import load_cifar10, load_cifar10c
from robustbench.utils import clean_accuracy
from robustbench.utils import load_model 
import argparse
import os
import torch
from autoattack import AutoAttack
import numpy as np
from torch import nn
import math

parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
args, _ = parser.parse_known_args()
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model_names = [
    ('Diffenderfer2021Winning_LRR', 'corruptions'),
]

for (model_name, t_type) in model_names:
    print(model_name)
    model = load_model(model_name=model_name, model_dir='./ckpt', dataset='cifar10', threat_model=t_type)
    model = model.cuda()

    # attacks_to_run = ['apgd-ce', 'apgd-dlr', 'fab', 'square', 'apgd-t', 'fab-t']
    adversary = AutoAttack(model, norm='Linf', eps=12/255, version='custom', attacks_to_run=['apgd-dlr'])
    adversary.apgd.n_restarts = 1

    #todo load data
    num_sample = 50000
    x_test, y_test = load_cifar10(n_examples=num_sample, data_dir='./data')

    #todo generate adv
    batch_size = 128
    x_adv = adversary.run_standard_evaluation(x_test, y_test.long(), bs=batch_size)

    #todo eval on samples
    acc = clean_accuracy(model, x_test, y_test, device=DEVICE)
    adv_acc = clean_accuracy(model, x_adv, y_test, device=DEVICE)
    print(f'Model: {model_name}, CIFAR-10 clean accuracy: {acc:.1%}')
    print(f'Model: {model_name}, CIFAR-10 adv accuracy: {adv_acc:.1%}')

    #todo save results
    save_path = os.path.join("./data/adv", model_name, adversary.__class__.__name__)
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, "adv.npy"), x_adv.permute(0, 2, 3, 1).detach().cpu().numpy())
    np.save(os.path.join(save_path, "label.npy"), y_test.detach().cpu().numpy())


if __name__ == '__main__':
    pass
