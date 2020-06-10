import numpy as np
import torch
import matplotlib.pyplot as plt
from pytorch_msssim import ssim

def clip_image(img, pixel_min=0, pixel_max=255):
    return torch.clamp(img, pixel_min, pixel_max)

def clip_image_perturb(img, img_org, perturb, model, epsilon, DEVICE, pixel_min=0, pixel_max=1):
    #len_perturb = torch.sum(torch.abs(perturb), dim=[1,2,3], keepdim=True)
    #perturb_normed = epsilon * perturb / len_perturb            # perturbation with same length epsilon
    switch = (torch.abs(img-img_org+perturb) > epsilon).float()     # indicate whether exceeds epsilon
    img_adv = switch*(img_org + epsilon) + (1.0-switch)*(img + perturb)
    img_adv = torch.clamp(img_adv, pixel_min, pixel_max)

    # Only keep samples whose predictions are not changed
    preds_org = model(img)
    preds_new = model(img_adv)
    _, preds_org = torch.max(preds_org.data, 1)
    _, preds_new = torch.max(preds_new.data, 1)
    switch = (preds_org == preds_new).float()
    switch = switch.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    img_adv = switch*img_adv + (1-switch)*img + (1-switch)*torch.randn(img.shape).to(DEVICE)*1e-6

    diff = img_adv - img_org
    return img_adv, diff

def normalize_dim_max_mag(tensor_in, MAG):
    # normalize each instance, so that its maximum abs(entry) is MAG
    max_entries = torch.max(torch.flatten(torch.abs(tensor_in), start_dim=1), dim=1, keepdim=True)[0] + 1e-6
    max_entries = max_entries.unsqueeze(-1)
    max_entries = max_entries.unsqueeze(-1)
    tensor_out = tensor_in / max_entries * MAG
    return tensor_out

def perturb_tangent(intp_rep, images_rep, MAG, DEVICE):
    # normalize gradient length
    intp_rep = intp_rep / torch.sqrt(torch.sum(intp_rep ** 2, dim=[2, 3], keepdim=True))
    # perturbation noise
    noise = ((torch.rand(images_rep.shape) - 0.5) * 2 * MAG).to(DEVICE)
    # noise perpendicular to gradient
    noise_perpend = noise - torch.sum(noise * intp_rep, dim=[2, 3], keepdim=True) * intp_rep
    images_aug = images_rep + noise_perpend
    return images_aug

def evaluation(model, dataloader, DEVICE):
    total, correct = 0, 0

    # keep the network in evaluation mode
    model.eval()
    for data in dataloader:
        inputs, labels = data
        # move the inputs and labels to gpu
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    return 100 * correct / total

def evaluation_aug(model, dataloader, DEVICE, explainer, n_intp_iter, n_rep, MAG):
    total, correct = 0, 0

    # keep the network in evaluation mode
    model.eval()
    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        # augmentation and perturbation
        smooth_intp = explainer.generate_smoothgrad(inputs, labels, DEVICE, n_intp_iter, MAG)
        inputs_rep = inputs.repeat(n_rep, 1, 1, 1)
        labels_aug = labels.repeat(n_rep)
        intp_rep = smooth_intp.repeat(n_rep, 1, 1, 1)
        inputs_aug = perturb_tangent(intp_rep, inputs_rep, MAG, DEVICE)
        # predict and evaluation
        outputs = model(inputs_aug)
        _, pred = torch.max(outputs.data, 1)
        total += labels_aug.size(0)
        correct += (pred == labels_aug).sum().item()
    return 100 * correct / total

def topk_intersection(intps1, intps2, K=50):
    intps1 = torch.flatten(intps1, start_dim=1)
    intps2 = torch.flatten(intps2, start_dim=1)
    order1 = torch.argsort(intps1, dim=1, descending=True).detach().cpu().numpy()[:, 0:K]
    order2 = torch.argsort(intps2, dim=1, descending=True).detach().cpu().numpy()[:, 0:K]
    avg_intersect = 0.
    for i in range(intps1.shape[0]):
        avg_intersect += float(np.intersect1d(order1[i], order2[i]).shape[0])/K
    avg_intersect /= intps1.shape[0]
    return avg_intersect

def rank_correlation(intps1, intps2):
    intps1 = torch.flatten(intps1, start_dim=1)
    intps2 = torch.flatten(intps2, start_dim=1)
    inds1_ranked = torch.argsort(intps1, dim=1, descending=True).detach().cpu().numpy()
    inds2_ranked = torch.argsort(intps2, dim=1, descending=True).detach().cpu().numpy()
    avg_corr =0.
    len_arr = intps1.shape[1]
    for i in range(intps1.shape[0]):
        arr_rank_1, arr_rank_2 = np.zeros(len_arr), np.zeros(len_arr)
        for j in range(len_arr):
            arr_rank_1[inds1_ranked[i,j]] = j
            arr_rank_2[inds2_ranked[i,j]] = j
        avg_corr += 1 - 6 * np.sum((arr_rank_2-arr_rank_1)**2) / (len_arr * (len_arr**2-1))
    avg_corr /= intps1.shape[0]
    return avg_corr

def ssim_similarity(inpts1, inpts2):
    inpts1 = inpts1.repeat(1,3,1,1)
    inpts2 = inpts2.repeat(1,3,1,1)
    #inpts1 -= inpts1.min(0, keepdim=True)[0]
    #inpts1 /= inpts1.max(0, keepdim=True)[0]
    #inpts2 -= inpts2.min(0, keepdim=True)[0]
    #inpts2 /= inpts2.max(0, keepdim=True)[0]
    ssim_result = ssim(inpts1, inpts2, data_range=1, size_average=True)
    return ssim_result

def show_2times2(imgs, imgs_new):
    fig = plt.figure(figsize=(8, 8))

    fig.add_subplot(2, 2, 1)
    img1 = np.squeeze(np.transpose(imgs.detach().cpu().numpy()[0], (1, 2, 0)))
    plt.imshow(img1)

    fig.add_subplot(2, 2, 2)
    img1_ = np.squeeze(np.transpose(imgs_new.detach().cpu().numpy()[0], (1, 2, 0)))
    plt.imshow(img1_)

    fig.add_subplot(2, 2, 3)
    img2 = np.squeeze(np.transpose(imgs.detach().cpu().numpy()[1], (1, 2, 0)))
    plt.imshow(img2)

    fig.add_subplot(2, 2, 4)
    img2_ = np.squeeze(np.transpose(imgs_new.detach().cpu().numpy()[1], (1, 2, 0)))
    plt.imshow(img2_)

    plt.show()


if __name__ == '__main__':
    intps1 = torch.from_numpy(np.array([106, 100, 86, 101, 99, 103, 97, 113, 112, 110])).unsqueeze(0)
    intps2 = torch.from_numpy(np.array([7, 27, 2, 50, 28, 29, 20, 12, 6, 17])).unsqueeze(0)
    print(rank_correlation(intps1, intps2))
    tensor_in = torch.from_numpy(np.random.rand(2,1,2,2))-0.5
    print(normalize_dim_max_mag(tensor_in, MAG=0.5))