import numpy as np
import time
import os
import argparse
import torch
import matplotlib.pyplot as plt
import torch.optim as optim

from data_fashmnist import data_fashmnist
from explainer_gradient import ExplainerGradient

labels_fashmnist = clothes = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
                              5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

def explain_random(model, loader, args, DEVICE, log_dir, model_dir):
    model.to(DEVICE)
    filename = os.path.join(model_dir, 'params.pt')
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(filename))
    else:
        model.load_state_dict(torch.load(filename, map_location='cpu'))
    explainer = None
    if args.intp == 'grad':
        explainer = ExplainerGradient(model)

    # iterating into the data
    dataiter = iter(loader)
    images, labels = dataiter.next()

    ID = 4
    img = images[ID:ID+4].to(DEVICE).requires_grad_()       # .requires_grad_()
    label = labels[ID:ID+4].to(DEVICE)
    print(label)

    # Visualization
    fig = plt.figure(figsize=(8, 8))
    # show clean image
    fig.add_subplot(2, 2, 1)
    img_show = np.squeeze(np.transpose(img.detach().cpu().numpy()[0], (1, 2, 0)))
    plt.imshow(img_show)

    # show clean explanation
    intp = explainer.generate_sensitivemap(img, label, DEVICE)
    intp_arr = intp.data.cpu().numpy()[0]
    intp_arr = np.transpose(intp_arr, (1, 2, 0))  # shape of the image should be height * width * channels
    intp_arr = intp_arr.clip(min=0)
    fig.add_subplot(2, 2, 2)
    plt.imshow(np.abs(np.squeeze(intp_arr)))  # (28, 28, 1) -> (28, 28)

    # show clean smoothed explanation
    intp = explainer.generate_smoothgrad(img, label, DEVICE, args.iter_smoothgrad, args.epsilon)
    intp = torch.abs(intp)
    intp_arr = intp.data.cpu().numpy()[0]
    intp_arr = np.transpose(intp_arr, (1, 2, 0))  # shape of the image should be height * width * channels
    intp_arr = intp_arr.clip(min=0)
    fig.add_subplot(2, 2, 3)
    plt.imshow(np.abs(np.squeeze(intp_arr)))  # (28, 28, 1) -> (28, 28)

    plt.show()

def explain_random(org_model, new_model, loader, args, DEVICE, log_dir, model_dir):
    org_model.to(DEVICE)
    new_model.to(DEVICE)

    if args.intp == 'grad':
        org_explainer = ExplainerGradient(org_model)
        new_explainer = ExplainerGradient(new_model)

    # iterating into the data
    dataiter = iter(loader)
    images, labels = dataiter.next()

    ID = 40
    img = images[ID:ID + 4].to(DEVICE).requires_grad_()  # .requires_grad_()
    label = labels[ID:ID + 4].to(DEVICE)
    print(label)

    # Visualization
    fig = plt.figure(figsize=(8, 8))
    # show clean image
    ax1 = fig.add_subplot(2, 2, 1)
    img_show = np.squeeze(np.transpose(img.detach().cpu().numpy()[0], (1, 2, 0)))
    ax1.imshow(img_show)
    ax1.title.set_text('Input image')

    # show clean explanation
    intp = org_explainer.generate_sensitivemap(img, label, DEVICE)
    intp_arr = intp.data.cpu().numpy()[0]
    intp_arr = np.transpose(intp_arr, (1, 2, 0))  # shape of the image should be height * width * channels
    intp_arr = intp_arr.clip(min=0)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(np.abs(np.squeeze(intp_arr)))  # (28, 28, 1) -> (28, 28)
    ax2.title.set_text('org model sensitive map')

    # show clean explanation
    intp = new_explainer.generate_sensitivemap(img, label, DEVICE)
    intp_arr = intp.data.cpu().numpy()[0]
    intp_arr = np.transpose(intp_arr, (1, 2, 0))  # shape of the image should be height * width * channels
    intp_arr = intp_arr.clip(min=0)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(np.abs(np.squeeze(intp_arr)))  # (28, 28, 1) -> (28, 28)
    ax3.title.set_text('ATEX model sensitive map')

    plt.show()
