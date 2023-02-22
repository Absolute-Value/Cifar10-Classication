import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.loader import dataset_loader

def test(args, model):
    device = args.device

    test_dataset = dataset_loader(train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=False)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    model.load_state_dict(torch.load(f'{args.result_dir}/model.pt'))

    results = torch.zeros((2, len(classes)), dtype=int)
    model.eval()
    loop = tqdm(test_loader)
    with torch.no_grad():
        for (inputs, labels) in loop:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            for pred, label in zip(preds, labels):
                results[1][int(label)] += 1
                if pred == label:
                    results[0][int(label)] += 1

    for (clas, correct, summ) in zip(classes, results[0], results[1]):
        print(f'{clas}: {correct}/{summ} ({correct/summ*100:.2f}%)')
    print(f'\nSUM: {results[0].sum()}/{results[1].sum()} ({results[0].sum()/results[1].sum()*100:.2f}%)')
