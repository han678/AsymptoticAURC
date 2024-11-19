import os
import sys

import torch

__all__ = ['PreResNet20', 'PreResNet20Drop', 'PreResNet56', 'PreResNet56Drop', 'PreResNet110', 'PreResNet110Drop', 'PreResNet164', 'PreResNet164Drop', 'PreResNet']

def get_network(args):
    """ return given network
    """
        # Determine number of classes and input size based on the dataset
    num_classes = {'cifar10': 10, 'cifar100': 100, 'svhn': 10, 'imagenet':1000}.get(args.dataset, 10)
    if args.arch == 'vgg16_bn':
        from models.vgg import vgg16_bn
        net = vgg16_bn(num_classes=num_classes)
    elif args.arch == 'vgg13_bn':
        from models.vgg import vgg13_bn
        net = vgg13_bn(num_classes=num_classes)
    elif args.arch == 'vgg11_bn':
        from models.vgg import vgg11_bn
        net = vgg11_bn(num_classes=num_classes)
    elif args.arch == 'vgg19_bn':
        from models.vgg import vgg19_bn
        net = vgg19_bn(num_classes=num_classes)
    elif args.arch == 'vgg11':
        from models.vgg import vgg11
        net = vgg11(num_classes=num_classes)
    elif args.arch == 'vgg13':
        from models.vgg import vgg13
        net = vgg13(num_classes=num_classes)
    elif args.arch == 'vgg16':
        from models.vgg import vgg16
        net = vgg16(num_classes=num_classes)
    elif args.arch == 'vgg19':
        from models.vgg import vgg19
        net = vgg19(num_classes=num_classes)
    elif args.arch == 'resnet18':
        from models.resnet import resnet18
        net = resnet18(num_classes=num_classes)
    elif args.arch == 'resnet34':
        from models.resnet import resnet34
        net = resnet34(num_classes=num_classes)
    elif args.arch == 'resnet50':
        from models.resnet import resnet50
        net = resnet50(num_classes=num_classes)
    elif args.arch == 'resnet101':
        from models.resnet import resnet101
        net = resnet101(num_classes=num_classes)
    elif args.arch == 'resnet152':
        from models.resnet import resnet152
        net = resnet152(num_classes=num_classes)
    elif args.arch == 'wrn':
        from models.wrn import WideResNet28x10
        net = WideResNet28x10(num_classes=num_classes)
    elif args.arch == 'preresnet20':    
        from models.preresnet import PreResNet20
        net = PreResNet20(num_classes=num_classes)
    elif args.arch == 'preresnet20drop':
        from models.preresnet import PreResNet20Drop
        net = PreResNet20Drop(num_classes=num_classes)
    elif args.arch == 'preresnet56':
        from models.preresnet import PreResNet56
        net = PreResNet56(num_classes=num_classes)
    elif args.arch == 'preresnet56drop':
        from models.preresnet import PreResNet56Drop
        net = PreResNet56Drop(num_classes=num_classes)
    elif args.arch == 'preresnet110':
        from models.preresnet import PreResNet110
        net = PreResNet110(num_classes=num_classes)
    elif args.arch == 'preresnet110drop':
        from models.preresnet import PreResNet110Drop
        net = PreResNet110Drop(num_classes=num_classes)
    elif args.arch == 'preresnet164':
        from models.preresnet import PreResNet164
        net = PreResNet164(num_classes=num_classes)
    elif args.arch == 'preresnet164drop':
        from models.preresnet import PreResNet164Drop
        net = PreResNet164Drop(num_classes=num_classes)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    return net