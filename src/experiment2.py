'''
This code mainly focus on experiment2
(1) synthesize attention maps of ATEX model
(2) compare to traditional methods such as smooth grad
(3) comparision metrics: Top-k intersection, Spearman’s rank order correlation
'''
import numpy as np
import time
import os
import argparse
import torch

from model_conv_fashmnist import ConvFashmnist, ConvCifar10, ConvMnist
from data_fashmnist import data_fashmnist, data_cifar10, data_mnist
from train import train
from advtrain_explain_tangval import advTrain_explanation_indirect, advTrain_FGSM
from explain import explain_random
from attack_explain import attack_explanation, attack_explanation_random
from compare_explain import compare_explanation

def main(args):
    name_exp = '%s-%s-%dE-%dB-%d' % (args.dataset, args.model, args.epochs, args.bsize, args.seed)
    DEVICE = "cpu"
    DEVICE = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else "cpu")
    log_dir = os.path.join('../results', name_exp)
    model_dir = os.path.join('../models', name_exp)

    # Build original model
    model_org = None
    model_new = None
    train_loader, valid_loader, test_loader = None, None, None
    if args.dataset == 'fashmnist' and args.model == 'conv':
        model_org = ConvFashmnist()
        model_new = ConvFashmnist()
        model_org_advfgsm = ConvFashmnist()
        model_new_advfgsm = ConvFashmnist()
        _, _, _, train_loader, valid_loader, test_loader = data_fashmnist(root='../data/fashmnist', args=args)
    elif args.dataset == 'cifar10' and args.model == 'conv':
        model_org = ConvCifar10()
        model_new = ConvCifar10()
        model_org_advfgsm = ConvCifar10()
        model_new_advfgsm = ConvCifar10()
        _, _, _, train_loader, valid_loader, test_loader = data_cifar10(root='../data/cifar10', args=args)
    elif args.dataset == 'mnist' and args.model == 'conv':
        model_org = ConvMnist()
        model_new = ConvMnist()
        model_org_advfgsm = ConvMnist()
        model_new_advfgsm = ConvMnist()
        _, _, _, train_loader, valid_loader, test_loader = data_mnist(root='../data/mnist', args=args)
    train(model_org, [train_loader, valid_loader, test_loader],
          args, DEVICE, log_dir, model_dir)

    
    # Test case: Get interpretation
    #explain_random(model_org, train_loader, args, DEVICE, log_dir, model_dir)
    # Test case: Adversarial attack on interpretation
    #::attack_explanation_random(model_org, train_loader, args, DEVICE, log_dir, model_dir)

    # Adversarial training to stabilize interpretation (ATEX)
    advTrain_explanation_indirect(model_org, model_new, [train_loader, valid_loader, test_loader],
                                  args, DEVICE, log_dir, model_dir, MAG=args.epsilon*4)

    
    org_filename = os.path.join(model_dir, args.dataset + '_params.pt')
    new_filename = os.path.join(model_dir, args.dataset + '_params_advintp.pt')

    if torch.cuda.is_available():
        model_org_advfgsm.load_state_dict(torch.load(org_filename))
        model_new_advfgsm.load_state_dict(torch.load(new_filename))
    else:
        model_org_advfgsm.load_state_dict(torch.load(org_filename, map_location='cpu'))
        model_new_advfgsm.load_state_dict(torch.load(new_filename, map_location='cpu'))
    
 
    # evaluate model adversairal defence performance
    advTrain_FGSM(model_org, model_org_advfgsm, [train_loader, valid_loader, test_loader],
                                  args, DEVICE, log_dir, model_dir)
    advTrain_FGSM(model_new, model_new_advfgsm, [train_loader, valid_loader, test_loader],
                  args, DEVICE, log_dir, model_dir)
    # explain_random(model_org, model_new, train_loader, args, DEVICE, log_dir, model_dir)

    

    '''
    # Evaluate model robustness before and after defense
    metric1_org, metric2_org = attack_explanation(model_org, [train_loader, valid_loader, test_loader],
                                    args, DEVICE, log_dir, model_dir, 'org')
    metric1_new, metric2_new = attack_explanation(model_new, [train_loader, valid_loader, test_loader],
                                    args, DEVICE, log_dir, model_dir, 'new')
    '''

    '''
    # compare model attention maps with SmoothGrad before and after defense
    metric1_org, metric2_org = compare_explanation(model_org, [train_loader, valid_loader, test_loader],
                                                  args, DEVICE, log_dir, model_dir, 'org')
    metric1_new, metric2_new = compare_explanation(model_new, [train_loader, valid_loader, test_loader],
                                                  args, DEVICE, log_dir, model_dir, 'new')
    '''
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0, help='Which gpu to use')
    parser.add_argument('--dataset', type=str, default='mnist', help='Which dataset to choose')
    parser.add_argument('--model', type=str, default='conv', help='Which model to choose')
    parser.add_argument('--epochs', default=2, type=int, help='number of total epochs to run')
    parser.add_argument('--bsize', default=100, type=int, help='mini-batch size')
    parser.add_argument('--lr', default=1.5e-3, type=float, help='learning rate')
    parser.add_argument('--train', default=False, type=bool, help='always train a new model or not')
    parser.add_argument('--intp', default='grad', type=str, help='type of interpretation method')
    parser.add_argument('--attack_type', default='topk', type=str, help='type of attacking method')
    parser.add_argument('--epsilon', default=0.08, type=float, help='perturbation range')
    parser.add_argument('--iter_along', default=2, type=int, help='iterations of moving along interpretation direction')
    parser.add_argument('--iter_attack', default=200, type=int, help='perturbation iterations')
    parser.add_argument('--perturb_step', default=0.02, type=float, help='perturbation step size')
    parser.add_argument('--iter_smoothgrad', default=10, type=int, help='smoothgrad iterations')
    parser.add_argument('--temperature', default=4, type=int, help='temperature for distillation')
    parser.add_argument('--K_intersection', default=50, type=int, help='topk intersection')
    parser.add_argument('--intp_attack_size', default=1000, type=int, help='instances set size considered in attack')
    parser.add_argument('--seed', default=1, type=int, help='seed for random number')
    parser.add_argument('--epsilon_smooth', default=0.1, type=float, help='epsilon for smoothgrad')
    parser.add_argument('--epsilon_FGSM', default=0.3, type=float, help='epsilon for FGSM')
    parser.add_argument('--FGSM_epochs', default=1, type=int, help='epoch for FGSM')
    parser.add_argument('--FGSM_number', default=1000, type=int, help='epoch for FGSM')
    args = parser.parse_args()

    main(args)

    # Notes
    # 1. Larger epsilon for defense increases interpretation robustness, but decreases model performance.
    # 2. Temperature needs to be tuned (4~5 for FashionMNIST).
