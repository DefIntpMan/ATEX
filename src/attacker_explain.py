import torch
import numpy as np
from torch.autograd import grad
from explainer_gradient import ExplainerGradient

class AttackerOfExplain():
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def attack_objective_targeted(self, intp, target_intp=np.arange(5)):
        obj_attack_toy = -intp[:, :, 10:15, 20:25].sum()    # random specification
        return obj_attack_toy

    def attack_objective_topk(self, intp, K=50):    # this objective is more formal
        thre = torch.sort(torch.flatten(intp, start_dim=1), dim=1, descending=True)[0][:, K]
        thre = thre.unsqueeze(-1)
        thre = thre.unsqueeze(-1)
        mask = (intp > thre).float()
        obj_attack = - (intp * mask).sum()
        return obj_attack

    def attack_objective_masscenter(self, intp, radius_ratio=0.05):
        pass

    def attack(self, input_images, target_labels, DEVICE,
               intp_type='grad', attack_type='targeted'):
        explainer = None
        # Choose the explanation method
        if intp_type == 'grad':
            explainer = ExplainerGradient(self.model)
        intp = explainer.generate_sensitivemap(input_images, target_labels, DEVICE, True)
        # Choose the attack objective
        obj_attak = None
        if attack_type == 'targeted':
            obj_attak = self.attack_objective_targeted(intp)
        if attack_type == 'topk':
            obj_attak = self.attack_objective_topk(intp)
        perturb = grad(obj_attak, input_images)[0]
        # Normalize
        max_perturb = torch.max(torch.flatten(torch.abs(perturb), start_dim=1), dim=1, keepdim=True)[0] + 1e-6
        max_perturb = max_perturb.unsqueeze(-1)
        max_perturb = max_perturb.unsqueeze(-1)
        perturb = perturb / max_perturb
        return perturb




