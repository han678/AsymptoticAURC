from __future__ import print_function

import argparse
import os

import torch.nn.functional as F
import models as models
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from models.utils import get_network
from utils.data_loader import prepare_dataset
from utils.logger import Logger

model_names = [
    'alexnet', 'squeezenet1_0', 'squeezenet1_1', 'densenet121',
    'densenet169', 'densenet201', 'densenet201', 'densenet161',
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'wrn'
]

def train_args():
    """Setup and parse common command line arguments."""
    parser = argparse.ArgumentParser(description='Training a classifier')
    parser.add_argument('--arch', '-a', metavar='ARCH', default="vgg19_bn", choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: vgg16_bn)')
    parser.add_argument('-d', '--dataset', default='imagenet', choices=['cifar10', 'svhn', 'cifar100', 'imagenet'])
    parser.add_argument('-j', '--workers', default=1, type=int)
    parser.add_argument('--loss', default='ce', type=str)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--train-batch', default=64, type=int)
    parser.add_argument('--test-batch', default=200, type=int)
    parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float)
    parser.add_argument('--epochdecay', default=30, type=int, help='number of epochs')
    
    parser.add_argument('-s', '--save', default='./outputs', type=str)
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true')
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--load_checkpoint', default=1, type=int)
    return parser.parse_args()


def test(net, test_loader, criterion):
    """Evaluate network on the given test dataset."""
    net.eval()
    test_loss = 0.0
    total_correct_1 = 0
    total_correct_5 = 0
    device = next(net.parameters()).device
    with torch.no_grad():
        for _, (images, targets) in enumerate(test_loader):
            images, targets = images.to(device), targets.to(device)
            outputs = net(images)
            loss = criterion(outputs, targets)
            logits = F.softmax(outputs, dim=1)
            _, pred = logits.topk(5, 1, largest=True, sorted=True)
            label = targets.view(targets.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()
            test_loss += loss.item()
            total_correct_1 += correct[:, :1].sum()
            total_correct_5 += correct[:, :5].sum()
    top1_acc = 100. * total_correct_1 / len(test_loader.dataset)
    top5_acc = 100. * total_correct_5 / len(test_loader.dataset)
    return test_loss, top1_acc, top5_acc


def train(net, train_loader, criterion, optimizer):
    """Train the network for one epoch."""
    net.train()
    train_loss = 0
    total_correct_1 = 0
    total_correct_5 = 0
    total = 0
    device = next(net.parameters()).device
    for batch_idx, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            logits = F.softmax(outputs, dim=1)
            _, pred = logits.topk(5, 1, largest=True, sorted=True)
            label = targets.view(targets.size(0), -1).expand_as(pred)
        correct = pred.eq(label).float()
        train_loss += loss.data.item()
        total_correct_1 += correct[:, :1].sum()
        total_correct_5 += correct[:, :5].sum()
        total += targets.size(0)
        
        top1_acc = 100. * float(total_correct_1) / float(total)
        top5_acc = 100. * float(total_correct_5) / float(total)
        avg_loss = train_loss / (batch_idx + 1)
        if batch_idx % 50 == 0:
            print('Train Set: {}, {} Loss: {:.2f} | Acc: {:.2f}% ({}/{}) '.format(
                batch_idx, len(train_loader), avg_loss, top1_acc, total_correct_1, total))
    return avg_loss, top1_acc, top5_acc


def train_base_model(args):
    """Train a base model on the given dataset."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    save_path = args.save + "/" + args.dataset
    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok = True)

    print('==> Preparing dataset %s' % args.dataset)
    trainloader, testloader  = prepare_dataset(
        args.dataset, args.train_batch, args.test_batch, args.workers
    )

    print("==> Creating model '%s'" % args.arch)
    model = get_network(args)
    
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model.cuda())
        cudnn.benchmark = True
    if args.load_checkpoint: 
        model_path = os.path.join(save_path, f'{args.arch}.pth')
        model.load_state_dict(torch.load(model_path))

    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9,
                          weight_decay=5e-4)

    title = f"{args.dataset}-{args.arch}"
    logger = Logger(os.path.join(save_path, f'{args.arch}_log.txt' if not args.evaluate else f'{args.arch}_eval.txt'),
                    title=title)
    
    add_names = ["Train Top1 Acc", "Train Top5 Acc", "Test Top1 Acc", "Test Top5 Acc"] if args.dataset in ['cifar100', 'imagenet'] else ["Train Acc", "Test Acc"]
    logger.set_names(['Epoch', 'Learning Rate', 'Train Loss', 'Test Loss'] + add_names)

    if not args.evaluate:
        lr = args.lr
        best_test_loss = float('inf')
        for epoch in range(1, args.epochs + 1):
            if epoch % args.epochdecay == 0:
                lr = lr / 5.0
                to_train = list(filter(lambda p: p.requires_grad, model.parameters()))
                optimizer = optim.SGD(to_train, lr=lr, momentum=0.9, weight_decay=5e-4)
            # train the model
            train_loss, train_acc_1, train_acc_5 = train(model, trainloader, criterion, optimizer)
            test_loss, test_acc_1, test_acc_5 = test(model, testloader, criterion)
            add_res = [train_acc_1, train_acc_5, test_acc_1, test_acc_5] if args.dataset in ['cifar100', 'imagenet'] else [train_acc_1, test_acc_1]
            logger.append([int(epoch), optimizer.param_groups[0]['lr'], train_loss, test_loss]+add_res)

            # Save the model with the best test loss
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model_path = os.path.join(save_path, f'{args.arch}.pth')
                torch.save(model.state_dict(), best_model_path)
                print(f'Saved new best model with test loss {best_test_loss:.4f} at epoch {epoch + 1}')
    else:
        model_path = os.path.join(save_path, f'{args.arch}.pth')
        if model_path is None:
            print("Model path not provided")
            return
        else:
            print("==> Load pretrained model '%s'" % args.arch)
            model.load_state_dict(torch.load(model_path))
            results = test(model, testloader, criterion)
            print(results)
            logger.append([0, None, None] + list(results))

    logger.close()


if __name__ == '__main__':
    args = train_args()
    train_base_model(args)
