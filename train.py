import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.loader import dataset_loader

def train(args, model):
    device = args.device

    train_dataset, val_dataset = dataset_loader()

    train_loader =  torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=False)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    for epoch in range(1, args.epochs+1):
        model.train()
        loop = tqdm(train_loader, desc=f'Train (epoch {epoch}): ')
        losses = []
        correct, num = 0, 0
        for inputs, labels in loop:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            correct += (torch.argmax(outputs, dim=1)==labels).sum().item()
            num += len(labels)

        train_losses.append(np.average(losses))
        train_accs.append(float(correct/num))
        print(f'Train ({epoch}/{args.epochs}) loss: {train_losses[-1]}, acc: {train_accs[-1]}')

        model.eval()
        loop = tqdm(val_loader, desc=f'Val (epoch {epoch}): ')
        losses = []
        correct, num = 0, 0
        with torch.no_grad():
            for inputs, labels in loop:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                losses.append(loss.item())
                correct += (torch.argmax(outputs, dim=1)==labels).sum().item()
                num += len(labels)

        val_losses.append(np.average(losses))
        val_accs.append(float(correct/num))
        print(f'Val ({epoch}/{args.epochs}) loss: {val_losses[-1]}, acc: {val_accs[-1]}')

    torch.save(model.state_dict(), f'{args.result_dir}/model.pt')
    print(f'Model exported.')

    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.ylim([0,2])
    plt.savefig(os.path.join(args.result_dir, 'loss.png'))
    plt.clf()

    plt.plot(train_accs)
    plt.plot(val_accs)
    plt.ylim([0,1])
    plt.savefig(os.path.join(args.result_dir, 'acc.png'))