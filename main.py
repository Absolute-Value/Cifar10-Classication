import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from utils.model import CNN
from train import train
from test import test

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    model = CNN(dim=args.dim).to(args.device)
    cuda_count = torch.cuda.device_count()
    print(cuda_count)
    if cuda_count > 1:
        model = nn.DataParallel(model, device_ids=list(range(cuda_count)))
    if args.train_skip:
        train(args, model)
    test(args, model)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=999)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--dim', type=int, default=128)
parser.add_argument('--train_skip', action='store_false')
parser.add_argument('--result_dir', type=str, default='outputs')
args = parser.parse_args()

os.makedirs(args.result_dir, exist_ok=True)
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:'+args.device)

main(args)
