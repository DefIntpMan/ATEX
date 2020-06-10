import numpy as np
import time
import os
import argparse
import torch
import matplotlib.pyplot as plt
from data_fashmnist import data_fashmnist

labels_fashmnist = clothes = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
                              5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

def vis_random(train_loader, valid_loader, test_loader, label_names):
    # iterating into the data
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    print(images.shape)  # shape of all 4 images

    ID = 4
    img = images[ID]
    print(img.shape)  # shape of one image
    print(labels[ID].item(), ': ', label_names[labels[ID].item()])  # label number

    img = img.numpy()
    img = np.transpose(img, (1, 2, 0))      # shape of the image should be height * width * channels

    plt.figure(figsize=(2, 2))
    plt.imshow(np.squeeze(img))     # (28, 28, 1) -> (28, 28)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0, help='Which gpu to use')
    parser.add_argument('--dataset', type=str, default='fashmnist', help='Which dataset to choose')
    parser.add_argument('--model', type=str, default='conv', help='Which model to choose')
    parser.add_argument('--epochs', default=15, type=int, help='number of total epochs to run')
    parser.add_argument('--bsize', default=100, type=int, help='mini-batch size')
    parser.add_argument('--lr', default=1.5e-3, type=float, help='learning rate')
    parser.add_argument('--train', default=False, type=bool, help='train a new model or not')
    parser.add_argument('--intp', default='grad', type=str, help='type of interpretation method')
    parser.add_argument('--seed', default=1, type=int, help='seed for random number')
    args = parser.parse_args()

    name_exp = '%s-%s-%dE-%dB-%d' % (args.dataset, args.model, args.epochs, args.bsize, args.seed)
    DEVICE = torch.device('cuda:{}'.format(args.device))
    log_dir = os.path.join('../results/logs', name_exp)
    model_dir = os.path.join('../models/exps', name_exp)

    train_loader, valid_loader, test_loader, label_names = None, None, None, None
    if args.dataset == 'fashmnist' and args.model == 'conv':
        _, _, _, train_loader, valid_loader, test_loader = data_fashmnist(root='../data/fashmnist', args=args)
        label_names = labels_fashmnist

    vis_random(train_loader, valid_loader, test_loader, label_names)