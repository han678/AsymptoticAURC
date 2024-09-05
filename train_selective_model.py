from __future__ import print_function
from numpy import test
import numpy as np  
import argparse, os, random, torch
import torch.backends.cudnn as cudnn, torch.nn as nn
import torch.optim as optim, torchvision.transforms as transforms
import models as models
from models.utils import get_network
from train_base_model import prepare_dataset, train
from utils import Logger
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data
from utils.loss import AURCLoss, ECELoss, SeleLoss, get_score_function
from utils.metrics import alpha_AURC, get_brier_score, geifman_AURC, get_ece_score
import torch.nn.functional as F

model_names = [
    'alexnet', 'squeezenet1_0', 'squeezenet1_1', 'densenet121',
    'densenet169', 'densenet201', 'densenet201', 'densenet161',
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'wrn'
]

def finetune_args():
    """Setup and parse common command line arguments."""
    parser = argparse.ArgumentParser(description='Training a selective classifier')
    parser.add_argument('--arch', '-a', metavar='ARCH', default="vgg16_bn", choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: vgg16_bn)')
    parser.add_argument('-d', '--dataset', default='cifar10', choices=['cifar10', 'svhn', 'cifar100', 'imagenet'])
    parser.add_argument('-j', '--workers', default=8, type=int)
    parser.add_argument('--loss', default='ce', type=str)
    parser.add_argument('--loss_type', default="sele", type=str, choices=["aurc", "ce", "sele", "ece"])
    parser.add_argument('--use_approx_aurc', default=1, type=int)
    parser.add_argument('--score_function', default="l2_norm", choices = ["softmax","neg_entropy","l2_norm"], type=str)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--train-batch', default=128, type=int)
    parser.add_argument('--test-batch', default=200, type=int)
    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float)
    parser.add_argument('--epochdecay', default=30, type=int, help='number of epochs')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true')
    parser.add_argument('--model_dir', default=None, type=str, help='path to the folder that contains pretrained model')
    parser.add_argument('--seed', default=42, type=int)
    return parser.parse_args()

def zero_one_loss(predictions, targets):
    return np.argmax(predictions, 1) != np.argmax(targets, 1)


def cross_entropy_loss(logits, targets):
    """Compute cross-entropy loss between prediction probabilities and targets."""
    ce_loss = -np.sum(targets * np.log(logits), axis=1) 
    return ce_loss


def evaluate_model(model, test_loader, score_func="softmax"):
    """Evaluate the model on the test set."""
    model.eval()
    all_logits, all_targets, all_confidences = [], [], []
    total_correct_1 = 0
    total_correct_5 = 0
    device = next(model.parameters()).device
    score_func = get_score_function(score_func)
    with torch.no_grad():
        for _, (images, targets) in enumerate(test_loader):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            logits = F.softmax(outputs, dim=1)
            confidence = score_func(outputs)
            _, pred = logits.topk(5, 1, largest=True, sorted=True)
            correct = pred.eq(targets.view(targets.size(0), -1).expand_as(pred)).float()
            total_correct_1 += correct[:, :1].sum().item()
            total_correct_5 += correct[:, :5].sum().item()
            all_logits.append(logits.cpu().numpy())
            all_targets.append(F.one_hot(targets, num_classes=logits.shape[1]).to(device).cpu().numpy())
            all_confidences.append(confidence.cpu().numpy())
    all_logits = np.vstack(all_logits)
    all_targets = np.vstack(all_targets)
    all_confidences = np.hstack(all_confidences)
    loss1 = zero_one_loss(all_logits, all_targets)
    loss2 = cross_entropy_loss(all_logits, all_targets)
    top1_acc = 100. * total_correct_1 / len(test_loader.dataset)
    top5_acc = 100. * total_correct_5 / len(test_loader.dataset)
    brier_score = get_brier_score(all_logits, all_targets)
    result = {"test_acc_1": top1_acc, "test_acc_5": top5_acc}
    aurc_result = geifman_AURC(residuals=loss1, confidence=all_confidences)
    result.update(aurc_result)
    result['ece'] = get_ece_score(all_logits, all_targets, n_bins=15)
    result["brier_score"] = brier_score
    result["0_1_exact_aurc"] = alpha_AURC(residuals=loss1, confidence=all_confidences, approx=False, return_dict=False)
    result["0_1_approx_aurc"] = alpha_AURC(residuals=loss1, confidence=all_confidences, approx=True, return_dict=False)
    result["exact_aurc"] = alpha_AURC(residuals=loss2, confidence=all_confidences, approx=False, return_dict=False)
    result["approx_aurc"] = alpha_AURC(residuals=loss2, confidence=all_confidences, approx=True, return_dict=False)
    return result

def train_selective_model(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print('==> Preparing dataset %s' % args.dataset)
    trainloader, testloader = prepare_dataset(
        args.dataset, args.train_batch, args.test_batch, args.workers
    )

    if args.model_dir is None:
        save_path = f"outputs/{args.dataset}" 
    else:
        save_path = args.model_dir
    model_path = os.path.join(save_path, f'{args.arch}.pth')
    if args.loss_type == "aurc":
        root_path = save_path + f"/{args.score_function}/approx" if args.use_approx_aurc else save_path + f"/{args.score_function}/exact"
    elif args.loss_type == "ce":
        root_path = save_path + "/ce"
    output_path = root_path + "/seed" + str(args.seed) + "lr" + str(args.lr)
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok = True)
    if model_path is None:
        print("Model path not provided")
        return
    else:
        print("==> Load pretrained model '%s'" % args.arch)
        model = get_network(args)   
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model.cuda())
            cudnn.benchmark = True
        model.load_state_dict(torch.load(model_path))
        results = evaluate_model(model, testloader)
        # print(results)
    if args.loss_type == "aurc":
        base_criterion = nn.CrossEntropyLoss()
        criterion = AURCLoss(criterion=base_criterion, score_func=args.score_function,  approx=args.use_approx_aurc)
    elif args.loss_type == "ce":
        criterion = nn.CrossEntropyLoss()
    elif args.loss_type == "sele":
        base_criterion = nn.CrossEntropyLoss()
        criterion = SeleLoss(criterion=base_criterion, score_func=args.score_function)
    elif args.loss_type == "ece":
        criterion = ECELoss(p=1, n_bins=15)
    else:
        raise ValueError("Invalid loss type")
    
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9,
                          weight_decay=5e-4)

    title = f"{args.dataset}-{args.arch}"
    logger = Logger(os.path.join(output_path, f'{args.arch}_bs{args.train_batch}_fintune.txt' if not args.evaluate else f'{args.arch}_bs{args.train_batch}_finetune_eval.txt'), title=title)
    add_names = ["Train Top1 Acc", "Train Top5 Acc", "Test Top1 Acc", "Test Top5 Acc"] if args.dataset in ['cifar100', 'imagenet'] else ['Train Acc', "Test Acc"]
    logger.set_names(['Epoch', 'Train Loss']+add_names+['AUC', 'EAURC', 'AURC(g)', "ECE", 'Brier Score', '0/1 AURC(e)', '0/1 AURC(a)', 'AURC(e)', 'AURC(a)'])
    model_info = [0, None, None, None] + list(results.values()) if args.dataset in ['cifar100', 'imagenet'] else [0, None, None] + list(results.values())[:1] + list(results.values())[2:]
    logger.append(model_info)
    lr = args.lr
    for epoch in range(1, args.epochs + 1):
        if epoch % args.epochdecay == 0:
            lr = lr / 5.0
            to_train = list(filter(lambda p: p.requires_grad, model.parameters()))
            optimizer = optim.SGD(to_train, lr=lr, momentum=0.9, weight_decay=5e-4)

        # Train the model
        train_loss, train_acc_1, train_acc_5 = train(model, trainloader, criterion, optimizer)
        results = evaluate_model(model, testloader)
        add_res = [train_acc_5] + list(results.values()) if args.dataset in ['cifar100', 'imagenet'] else list(results.values())[:1] + list(results.values())[2:] # remove the top 5 accuracy
        logger.append([int(epoch), train_loss, train_acc_1]+ add_res)

        # Save the model every 10 epochs
        if epoch == args.epochs:
            model_path = os.path.join(output_path, f'finetune_{args.arch}_bs{args.train_batch}_{epoch}.pth')
            torch.save(model.state_dict(), model_path)
            print(f'Saved model checkpoint at epoch {epoch} with test acc {results["test_acc_1"]:.4f}')

    logger.close()

if __name__ == '__main__':
    args = finetune_args()
    train_selective_model(args)
