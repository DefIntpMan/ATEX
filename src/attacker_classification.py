import torch
import numpy as np
from torch.autograd import grad
import torch.nn as nn

class AttackerOfClassification():
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def fgsm_attack(self, image, epsilon, data_grad):
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = image + epsilon * sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return perturbed_image

    def fgsm_test(self, model, device, test_loader, epsilon):

        # Accuracy counter
        correct = 0
        adv_examples = []
        criterion = nn.CrossEntropyLoss()

        # Loop over all examples in test set
        for data_batch in test_loader:
            datas, targets = data_batch
            for data, target in zip(datas, targets):
                # Send the data and label to the device
                data, target = data.to(device).requires_grad_(), target.to(device)
                data.requires_grad = True
                # Set requires_grad attribute of tensor. Important for Attack
                data = data.unsqueeze(0)
                target = target.unsqueeze(0)
                # Forward pass the data through the model
                output = model(data)
                init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

                # If the initial prediction is wrong, dont bother attacking, just move on
                if init_pred.item() != target.item():
                    continue

                # Calculate the loss
                loss = criterion(output, target)

                # Zero all existing gradients
                model.zero_grad()

                # Calculate gradients of model in backward pass
                data.retain_grad()
                loss.backward()

                # Collect datagrad
                data_grad = data.grad.data

                # Call FGSM Attack
                perturbed_data = self.fgsm_attack(data, epsilon, data_grad)

                # Re-classify the perturbed image
                output = model(perturbed_data)

                # Check for success
                final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                if final_pred.item() == target.item():
                    correct += 1

        # Calculate final accuracy for this epsilon
        final_acc = correct / float(len(test_loader))
        print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

        # Return the accuracy and an adversarial example
        return final_acc





