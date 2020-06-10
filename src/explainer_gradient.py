import torch
from torch.autograd import grad
from utils import clip_image

class ExplainerGradient():
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradients = None

    def generate_sensitivemap(self, input_images, target_labels, DEVICE, further_grad=False):
        model_output = self.model(input_images)
        self.model.zero_grad()
        gradients = grad(model_output[torch.arange(model_output.size()[0]), target_labels].sum(), input_images,
                         create_graph=further_grad)[0]  # the .sum() operation will not entangle different samples
        #gradients = torch.clamp(gradients, min=0)
        gradients = torch.abs(gradients)
        return gradients

    def generate_smoothgrad(self, input_images, target_labels, DEVICE, iter_smoothgrad,
                            MAG, further_grad=False):
        gradients = torch.zeros_like(input_images)
        for t in range(iter_smoothgrad):
            noise = torch.randn(input_images.shape) * MAG
            noise = noise.to(DEVICE)
            noisy_images = (input_images + noise).requires_grad_()
            noisy_images = clip_image(noisy_images)

            model_output = self.model(noisy_images)
            self.model.zero_grad()
            grad_t = grad(model_output[torch.arange(model_output.size()[0]), target_labels].sum(), noisy_images,
                             create_graph=further_grad)[0]
            gradients += grad_t
        gradients /= iter_smoothgrad
        return gradients


