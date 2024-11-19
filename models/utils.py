import os
import sys

def get_network(args):
    """ return given network
    """
        # Determine number of classes and input size based on the dataset
    num_classes = {'cifar10': 10, 'cifar100': 100, 'svhn': 10, 'imagenet':1000}.get(args.dataset, 10)
    if args.arch == 'VGG16BN':
        from models.vgg import VGG16BN
        net = VGG16BN(num_classes=num_classes)
    elif args.arch == 'VGG16':
        from models.vgg import VGG16
        net = VGG16(num_classes=num_classes)
    elif args.arch == 'VGG16Drop':
        from models.vgg import VGG16Drop
        net = VGG16Drop(num_classes=num_classes)
    elif args.arch == 'VGG16BNDrop':
        from models.vgg import VGG16BNDrop
        net = VGG16BNDrop(num_classes=num_classes)
    elif args.arch == 'VGG19BN':
        from models.vgg import VGG19BN
        net = VGG19BN(num_classes=num_classes)
    elif args.arch == 'VGG19':
        from models.vgg import VGG19
        net = VGG19(num_classes=num_classes)
    elif args.arch == 'VGG19Drop':
        from models.vgg import VGG19Drop
        net = VGG19Drop(num_classes=num_classes)
    elif args.arch == 'VGG19BNDrop':
        from models.vgg import VGG19BNDrop
        net = VGG19BNDrop(num_classes=num_classes)
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
    elif args.arch == 'PreResNet20':    
        from models.preresnet import PreResNet20
        net = PreResNet20(num_classes=num_classes)
    elif args.arch == 'PreResNet20Drop':
        from models.preresnet import PreResNet20Drop
        net = PreResNet20Drop(num_classes=num_classes)
    elif args.arch == 'PreResNet56':
        from models.preresnet import PreResNet56
        net = PreResNet56(num_classes=num_classes)
    elif args.arch == 'PreResNet56drop':
        from models.preresnet import PreResNet56Drop
        net = PreResNet56Drop(num_classes=num_classes)
    elif args.arch == 'PreResNet110':
        from models.preresnet import PreResNet110
        net = PreResNet110(num_classes=num_classes)
    elif args.arch == 'PreResNet110drop':
        from models.preresnet import PreResNet110Drop
        net = PreResNet110Drop(num_classes=num_classes)
    elif args.arch == 'PreResNet164':
        from models.preresnet import PreResNet164
        net = PreResNet164(num_classes=num_classes)
    elif args.arch == 'PreResNet164drop':
        from models.preresnet import PreResNet164Drop
        net = PreResNet164Drop(num_classes=num_classes)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    return net
