from robustbench.data import load_cifar10
from robustbench.utils import clean_accuracy
from robustbench.utils import load_model 
import argparse
import os

parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
args, _ = parser.parse_known_args()

model_names = [
    ('Diffenderfer2021Winning_LRR', 'corruptions'), 
]

for model_name, t_type in model_names:
    print(model_name)
    model = load_model(model_name=model_name, model_dir='./ckpt', dataset='cifar10', threat_model=t_type)

