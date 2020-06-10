import torchvision
import torchvision.transforms as transforms
import torch


def data_fashmnist(root, args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    trainset = torchvision.datasets.FashionMNIST(root=root, train=True, download=True,
                                                 transform=transforms.ToTensor())
    num_train = int(trainset.__len__()*5/6)
    trainset, validset = torch.utils.data.random_split(trainset, [num_train, trainset.__len__()-num_train])
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.bsize, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=args.bsize, shuffle=True)
    testset = torchvision.datasets.FashionMNIST(root=root, train=False, download=True,
                                                transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.bsize, shuffle=False)

    print("Train, Val, Test sizes: %d, %d, %d" % (trainset.__len__(), validset.__len__(), testset.__len__()))

    return trainset, validset, testset, train_loader, valid_loader, test_loader

def data_cifar10(root, args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    transform_test = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True,
                                            transform=transform_train)
    num_train = int(trainset.__len__() * 5 / 6)
    trainset, validset = torch.utils.data.random_split(trainset, [num_train, trainset.__len__() - num_train])
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.bsize,shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=args.bsize, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.bsize, shuffle=False)
    print("Train, Val, Test sizes: %d, %d, %d" % (trainset.__len__(), validset.__len__(), testset.__len__()))
    return trainset, validset, testset, train_loader, valid_loader, test_loader

def data_mnist(root, args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    trainset = torchvision.datasets.MNIST(root=root, train=True, download=True,
                                          transform=transforms.ToTensor())
    num_train = int(trainset.__len__() * 5 / 6)
    trainset, validset = torch.utils.data.random_split(trainset, [num_train, trainset.__len__() - num_train])
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.bsize, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=args.bsize, shuffle=True)
    testset = torchvision.datasets.MNIST(root=root, train=False, download=True,
                                         transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.bsize, shuffle=False)

    print("Train, Val, Test sizes: %d, %d, %d" % (trainset.__len__(), validset.__len__(), testset.__len__()))
    return trainset, validset, testset, train_loader, valid_loader, test_loader
