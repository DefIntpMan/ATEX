# Samples along the perpendicular direction should have the same prediction score.
# Samples along the gradient direction should have varied prediction scores.
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import evaluation, evaluation_aug, clip_image, perturb_tangent, show_2times2, normalize_dim_max_mag
from explainer_gradient import ExplainerGradient

def advTrain_explanation_indirect(model_org, model_new, loaders, args, DEVICE, log_dir, model_dir,
                                  MAG, n_rep=16):
    train_loader, valid_loader, test_loader = loaders
    model_org.to(DEVICE)
    model_new.to(DEVICE)
    criterion = nn.KLDivLoss()
    learning_rate = args.lr
    optimizer = optim.Adam(model_new.parameters(), lr=learning_rate)
    filename_load = os.path.join(model_dir, 'params.pt')
    filename_save = os.path.join(model_dir, 'params_advintp.pt')

    if os.path.exists(model_dir):
        model_new.load_state_dict(torch.load(filename_load))
        if True:
            acc_train = evaluation(model_org, train_loader, DEVICE)
            acc_valid = evaluation(model_org, valid_loader, DEVICE)
            acc_test = evaluation(model_org, test_loader, DEVICE)
            print('Original model loaded. Train acc: %0.2f, Valid acc: %0.2f, Test acc: %0.2f' %
                  (acc_train, acc_valid, acc_test))
            acc_train = evaluation(model_new, train_loader, DEVICE)
            acc_valid = evaluation(model_new, valid_loader, DEVICE)
            acc_test = evaluation(model_new, test_loader, DEVICE)
            print('New model loaded. Train acc: %0.2f, Valid acc: %0.2f, Test acc: %0.2f' %
                  (acc_train, acc_valid, acc_test))
    else:
        sys.exit("Pre-trained model does not exist! Terminate.")

    explainer = None
    if args.intp == 'grad':
        explainer = ExplainerGradient(model_new)

    acc_best = 0.0
    for epoch in range(args.epochs):
        model_org.eval()
        model_new.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE).requires_grad_()
            labels = labels.to(DEVICE)
            # Get smooth interpretation
            smooth_intp = explainer.generate_smoothgrad(images, labels, DEVICE,
                                                        args.iter_smoothgrad, args.epsilon)
            # Training batches along the perpendicular direction of interpretation
            perturb_along_max = normalize_dim_max_mag(smooth_intp, MAG*8)
            for t in range(args.iter_along):
                # 1. data copies, where inputs, outputs, interpretations are correspondent
                images_rep = images.repeat(n_rep, 1, 1, 1)
                softlabels_aug = F.softmax(outputs.repeat(n_rep, 1), dim=-1)
                intp_rep = smooth_intp.repeat(n_rep, 1, 1, 1)
                # 2. perturb instance randomly
                noises = ((torch.rand(images_rep.shape) - 0.5) * 2 * MAG).to(DEVICE)
                images_rep += noises
                images_aug = perturb_tangent(intp_rep, images_rep, MAG, DEVICE)
                # show
                #show_2times2(images_rep, images_aug)

                # 3. train this augmented batch
                images_aug = images_aug.detach()
                softlabels_aug = softlabels_aug.detach()
                optimizer.zero_grad()
                outputs_new = model_new(images_aug)
                loss = criterion(F.log_softmax(outputs_new, dim=-1), softlabels_aug)
                loss.backward()
                optimizer.step()

        acc_train = evaluation(model_new, train_loader, DEVICE)
        acc_valid = evaluation(model_new, valid_loader, DEVICE)
        acc_test = evaluation(model_new, test_loader, DEVICE)
        acc_valid_aug = evaluation_aug(model_new, valid_loader, DEVICE, explainer, args.iter_smoothgrad, n_rep, MAG)
        acc_test_aug = evaluation_aug(model_new, test_loader, DEVICE, explainer, args.iter_smoothgrad, n_rep, MAG)
        print('Epoch: %d/%d, Train acc: %0.2f, Valid acc: %0.2f, Test acc: %0.2f ; AugValid acc: %0.2f, AugTest acc: %0.2f' %
              (epoch, args.epochs, acc_train, acc_valid, acc_test, acc_valid_aug, acc_test_aug))
        torch.save(model_new.state_dict(), filename_save)


