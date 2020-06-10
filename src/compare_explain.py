'''
we compare ATEX interpretation with SmoothGard
'''
import numpy as np
import os
import sys
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from explainer_gradient import ExplainerGradient
from attacker_explain import AttackerOfExplain
from pytorch_msssim import ssim
from utils import evaluation, clip_image_perturb, topk_intersection, rank_correlation, ssim_similarity

def compare_explanation(model, loaders, args, DEVICE, log_dir, model_dir, flag):
    print("######### compare explain with smoothgrad ################")
    train_loader, valid_loader, test_loader = loaders
    model.to(DEVICE)
    if os.path.exists(model_dir):
        if flag == 'org':
            filename = os.path.join(model_dir, args.dataset + '_params.pt')
        else:
            filename = os.path.join(model_dir, args.dataset + '_params_advintp.pt')
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(filename))
        else:
            model.load_state_dict(torch.load(filename, map_location='cpu'))
        acc_train = evaluation(model, train_loader, DEVICE)
        acc_valid = evaluation(model, valid_loader, DEVICE)
        acc_test = evaluation(model, test_loader, DEVICE)
        if flag == 'org':
            print('Original model loaded. Train acc: %0.2f, Valid acc: %0.2f, Test acc: %0.2f' %
                  (acc_train, acc_valid, acc_test))
        else:
            print('New model loaded. Train acc: %0.2f, Valid acc: %0.2f, Test acc: %0.2f' %
                  (acc_train, acc_valid, acc_test))
    else:
        sys.exit("Pre-trained model does not exist! Terminate.")

    explainer = None
    if args.intp == 'grad':
        explainer = ExplainerGradient(model)

    metric1 = 0.
    metric2 = 0.
    metric3 = 0.

    n_iters = 0
    num_insts = 0
    for i, (images, labels) in tqdm(enumerate(test_loader)):
        images = images.to(DEVICE).requires_grad_()
        labels = labels.to(DEVICE)

        # evaluate similarity
        intp_sensitivemap = explainer.generate_sensitivemap(images, labels, DEVICE)
        intp_smoothgrad = explainer.generate_smoothgrad(images, labels, DEVICE, args.iter_smoothgrad, args.epsilon)

        metric1 += topk_intersection(intp_sensitivemap, intp_smoothgrad, args.K_intersection)
        metric2 += rank_correlation(intp_sensitivemap, intp_smoothgrad)
        metric3 += ssim_similarity(intp_sensitivemap, intp_smoothgrad)

        n_iters += 1
        num_insts += images.shape[0]
        if num_insts >= args.intp_attack_size:
            break
    metric1 /= n_iters
    metric2 /= n_iters
    metric3 /= n_iters
    print('Top-k intersection: %0.3f; Rank correlation: %0.3f; SSIM similarity: %0.3f' %
          (metric1, metric2, metric3))

    return metric1, metric2
