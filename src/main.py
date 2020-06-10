import numpy as np
import time
import os
import argparse
import torch

from model_conv_fashmnist import ConvFashmnist
from data_fashmnist import data_fashmnist
from train import train
from advtrain_explain_tangval import advTrain_explanation_indirect
from explain import explain_random
from attack_explain import attack_explanation, attack_explanation_random

def main(args):
    name_exp = '%s-%s-%dE-%dB-%d' % (args.dataset, args.model, args.epochs, args.bsize, args.seed)
    DEVICE = torch.device('cuda:{}'.format(args.device))
    log_dir = os.path.join('../results', name_exp)
    model_dir = os.path.join('../models', name_exp)

    # Build original model
    model_org = None
    model_new = None
    train_loader, valid_loader, test_loader = None, None, None
    if args.dataset == 'fashmnist' and args.model == 'conv':
        model_org = ConvFashmnist()
        model_new = ConvFashmnist()
        _, _, _, train_loader, valid_loader, test_loader = data_fashmnist(root='../data/fashmnist', args=args)

    train(model_org, [train_loader, valid_loader, test_loader],
          args, DEVICE, log_dir, model_dir)

    # Test case: Get interpretation
    # explain_random(model_org, train_loader, args, DEVICE, log_dir, model_dir)
    # Test case: Adversarial attack on interpretation
    # attack_explanation_random(model_org, train_loader, args, DEVICE, log_dir, model_dir)

    # Adversarial training to stabilize interpretation (ATEX)
    advTrain_explanation_indirect(model_org, model_new, [train_loader, valid_loader, test_loader],
                                  args, DEVICE, log_dir, model_dir, MAG=args.epsilon*4)

    # Evaluate model robustness before and after defense
    metric1_org, metric2_org = attack_explanation(model_org, [train_loader, valid_loader, test_loader],
                                    args, DEVICE, log_dir, model_dir, 'org')
    metric1_new, metric2_new = attack_explanation(model_new, [train_loader, valid_loader, test_loader],
                                    args, DEVICE, log_dir, model_dir, 'new')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0, help='Which gpu to use')
    parser.add_argument('--dataset', type=str, default='fashmnist', help='Which dataset to choose')
    parser.add_argument('--model', type=str, default='conv', help='Which model to choose')
    parser.add_argument('--epochs', default=2, type=int, help='number of total epochs to run')
    parser.add_argument('--bsize', default=100, type=int, help='mini-batch size')
    parser.add_argument('--lr', default=1.5e-3, type=float, help='learning rate')
    parser.add_argument('--train', default=False, type=bool, help='always train a new model or not')
    parser.add_argument('--intp', default='grad', type=str, help='type of interpretation method')
    parser.add_argument('--attack_type', default='topk', type=str, help='type of attacking method')
    parser.add_argument('--epsilon', default=0.02, type=float, help='perturbation range')
    parser.add_argument('--iter_along', default=11, type=int, help='iterations of moving along interpretation direction')
    parser.add_argument('--iter_attack', default=200, type=int, help='perturbation iterations')
    parser.add_argument('--perturb_step', default=0.002, type=float, help='perturbation step size')
    parser.add_argument('--iter_smoothgrad', default=10, type=int, help='smoothgrad iterations')
    parser.add_argument('--temperature', default=4, type=int, help='temperature for distillation')
    parser.add_argument('--K_intersection', default=50, type=int, help='topk intersection')
    parser.add_argument('--intp_attack_size', default=1000, type=int, help='instances set size considered in attack')
    parser.add_argument('--seed', default=1, type=int, help='seed for random number')
    args = parser.parse_args()

    main(args)

    # Notes
    # 1. Larger epsilon for defense increases interpretation robustness, but decreases model performance.
    # 2. Temperature needs to be tuned (4~5 for FashionMNIST).
