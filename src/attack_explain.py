# Note: The interpretation of a piece-wise linear model cannot be attacked using 2nd order gradients!!!
import numpy as np
import os
import sys
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from explainer_gradient import ExplainerGradient
from attacker_explain import AttackerOfExplain
from utils import evaluation, clip_image_perturb, topk_intersection, rank_correlation

def attack_explanation(model, loaders, args, DEVICE, log_dir, model_dir, flag):
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
    intp_attacker = AttackerOfExplain(model)

    metric1 = 0.
    metric2 = 0.
    n_iters = 0
    num_insts = 0
    for i, (images, labels) in tqdm(enumerate(test_loader)):
        images = images.to(DEVICE).requires_grad_()
        labels = labels.to(DEVICE)

        # attack
        images_adv = images + 0.0
        for t in range(args.iter_attack):
            perturb = intp_attacker.attack(images_adv, labels, DEVICE,
                                           attack_type=args.attack_type) * args.perturb_step
            images_adv, diff = clip_image_perturb(images_adv, images, perturb, model, args.epsilon, DEVICE)
            # if t == 50:    # check perturbation magnitude
            #     print(t)
            #     print(torch.max(torch.flatten(diff, start_dim=1), dim=1)[0])

        # evaluate attack
        intp_org = explainer.generate_sensitivemap(images, labels, DEVICE)
        intp_new = explainer.generate_sensitivemap(images_adv, labels, DEVICE)
        metric1 += topk_intersection(intp_org, intp_new, args.K_intersection)
        metric2 += rank_correlation(intp_org, intp_new)

        n_iters += 1
        num_insts += images.shape[0]
        if num_insts >= args.intp_attack_size:
            break
    metric1 /= n_iters
    metric2 /= n_iters
    if flag == 'org':
        print('Original model Top-k intersection: %0.3f; Rank correlation: %0.3f' %
          (metric1, metric2))
    else:
        print('New model Top-k intersection: %0.3f; Rank correlation: %0.3f' %
              (metric1, metric2))

    return metric1, metric2


def attack_explanation_random(model, loader, args, DEVICE, log_dir, model_dir):
    model.to(DEVICE)
    filename = os.path.join(model_dir, 'params.pt')
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(filename))
    else:
        model.load_state_dict(torch.load(filename, map_location='cpu'))

    # iterating into the data
    dataiter = iter(loader)
    images, labels = dataiter.next()

    ID = 4
    img = images[ID:ID+4].to(DEVICE).requires_grad_()  # .requires_grad_()
    label = labels[ID:ID+4].to(DEVICE)
    print(img[0,0,0])
    print(label)

    explainer = None
    if args.intp == 'grad':
        explainer = ExplainerGradient(model)

    intp_attacker = AttackerOfExplain(model)
    img_adv = img + 0.0
    for t in range(args.iter_attack):
        perturb = intp_attacker.attack(img_adv, label, DEVICE) * args.perturb_step
        img_adv, _ = clip_image_perturb(img_adv, img, perturb, model, args.epsilon, DEVICE)
        intp = explainer.generate_sensitivemap(img_adv, label, DEVICE)
        # print(intp_attacker.attack_objective_targeted(intp))

    # Visualization
    fig = plt.figure(figsize=(8, 8))
    # show clean image
    fig.add_subplot(2, 2, 1)
    img_show = np.squeeze(np.transpose(img.detach().cpu().numpy()[0], (1, 2, 0)))
    plt.imshow(img_show)

    # show clean explanation
    intp = explainer.generate_sensitivemap(img, label, DEVICE)
    intp_arr = intp.data.cpu().numpy()[0]         # show the first sample
    intp_arr = np.transpose(intp_arr, (1, 2, 0))  # shape of the image should be height * width * channels
    intp_arr = intp_arr.clip(min=0)
    fig.add_subplot(2, 2, 2)
    plt.imshow(np.abs(np.squeeze(intp_arr)))  # (28, 28, 1) -> (28, 28)

    # show perturbed image
    fig.add_subplot(2, 2, 3)
    img_adv_show = np.squeeze(np.transpose(img_adv.detach().cpu().numpy()[0], (1, 2, 0)))
    plt.imshow(img_adv_show)

    # show perturbed explanation
    intp = explainer.generate_sensitivemap(img_adv, label, DEVICE)
    intp_arr = intp.data.cpu().numpy()[0]
    intp_arr = np.transpose(intp_arr, (1, 2, 0))  # shape of the image should be height * width * channels
    intp_arr = intp_arr.clip(min=0)
    print(-intp[:, :, 10:15, 20:25].sum())
    fig.add_subplot(2, 2, 4)
    plt.imshow(np.abs(np.squeeze(intp_arr)))  # (28, 28, 1) -> (28, 28)

    plt.show()
