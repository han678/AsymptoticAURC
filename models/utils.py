import sys

def get_network(args):
    """ return given network
    """
        # Determine number of classes and input size based on the dataset
    num_classes = {'cifar10': 10, 'cifar100': 100, 'svhn': 10, 'imagenet':1000}.get(args.dataset, 10)
    if args.dataset =="imagenet":
        if args.arch == 'resnet18':
            from models.imagenet.resnet import resnet18
            net = resnet18(num_classes=num_classes)
        elif args.arch == 'resnet34':
            from models.imagenet.resnet import resnet34
            net = resnet34(num_classes=num_classes)
        elif args.arch == 'resnet50':
            from models.imagenet.resnet import resnet50
            net = resnet50(num_classes=num_classes)
        elif args.arch == 'resnet101':
            from models.imagenet.resnet import resnet101
            net = resnet101(num_classes=num_classes)
        elif args.arch == 'resnet152':
            from models.imagenet.resnet import resnet152
            net = resnet152(num_classes=num_classes)
        elif args.arch == 'vgg11':
            from models.imagenet.vgg import vgg11
            net = vgg11(num_classes=num_classes)
        elif args.arch == 'vgg11_bn':
            from models.imagenet.vgg import vgg11_bn
            net = vgg11_bn(num_classes=num_classes)    
        elif args.arch == 'vgg13':
            from models.imagenet.vgg import vgg13
            net = vgg13(num_classes=num_classes)
        elif args.arch == 'vgg13_bn':
            from models.imagenet.vgg import vgg13_bn
            net = vgg13_bn(num_classes=num_classes)
        elif args.arch == 'vgg16':
            from models.imagenet.vgg import vgg16
            net = vgg16(num_classes=num_classes)
        elif args.arch == 'vgg19':
            from models.imagenet.vgg import vgg19
            net = vgg19(num_classes=num_classes)
        elif args.arch == 'vgg16_bn':
            from models.imagenet.vgg import vgg16_bn
            net = vgg16_bn(num_classes=num_classes)
        elif args.arch == 'vgg19_bn':
            from models.imagenet.vgg import vgg19_bn
            net = vgg19_bn(num_classes=num_classes)
        else:
            print('the network name you have entered is not supported yet')
            sys.exit()
    else:
        if args.arch == 'vgg16_bn':
            from models.cifar.vgg import vgg16_bn
            net = vgg16_bn(num_classes=num_classes)
        elif args.arch == 'vgg13_bn':
            from models.cifar.vgg import vgg13_bn
            net = vgg13_bn(num_classes=num_classes)
        elif args.arch == 'vgg11_bn':
            from models.cifar.vgg import vgg11_bn
            net = vgg11_bn(num_classes=num_classes)
        elif args.arch == 'vgg19_bn':
            from models.cifar.vgg import vgg19_bn
            net = vgg19_bn(num_classes=num_classes)
        elif args.arch == 'vgg11':
            from models.cifar.vgg import vgg11
            net = vgg11(num_classes=num_classes)
        elif args.arch == 'vgg13':
            from models.cifar.vgg import vgg13
            net = vgg13(num_classes=num_classes)
        elif args.arch == 'vgg16':
            from models.cifar.vgg import vgg16
            net = vgg16(num_classes=num_classes)
        elif args.arch == 'vgg19':
            from models.cifar.vgg import vgg19
            net = vgg19(num_classes=num_classes)
        elif args.arch == 'alexnet':
            from models.cifar.alexnet import alexnet
            net = alexnet(num_classes=num_classes)
        elif args.arch == 'densenet121':
            from models.cifar.densenet import densenet121
            net = densenet121(num_classes=num_classes)
        elif args.arch == 'densenet161':
            from models.cifar.densenet import densenet161
            net = densenet161(num_classes=num_classes)
        elif args.arch == 'densenet169':
            from models.cifar.densenet import densenet169
            net = densenet169(num_classes=num_classes)
        elif args.arch == 'densenet201':
            from models.cifar.densenet import densenet201
            net = densenet201(num_classes=num_classes)
        elif args.arch == 'resnet18':
            from models.cifar.resnet import resnet18
            net = resnet18(num_classes=num_classes)
        elif args.arch == 'resnet34':
            from models.cifar.resnet import resnet34
            net = resnet34(num_classes=num_classes)
        elif args.arch == 'resnet50':
            from models.cifar.resnet import resnet50
            net = resnet50(num_classes=num_classes)
        elif args.arch == 'resnet101':
            from models.cifar.resnet import resnet101
            net = resnet101(num_classes=num_classes)
        elif args.arch == 'resnet152':
            from models.cifar.resnet import resnet152
            net = resnet152(num_classes=num_classes)
        elif args.arch == 'resnext50':
            from models.cifar.resnext import resnext50
            net = resnext50(num_classes=num_classes)
        elif args.arch == 'resnext101':
            from models.cifar.resnext import resnext101
            net = resnext101(num_classes=num_classes)
        elif args.arch == 'resnext152':
            from models.cifar.resnext import resnext152
            net = resnext152(num_classes=num_classes)
        elif args.arch == 'wrn':
            from models.cifar.wrn import WideResNet28x10
            net = WideResNet28x10(num_classes=num_classes)
        else:
            print('the network name you have entered is not supported yet')
            sys.exit()

    return net