import numpy as np
import torch
from torch import nn
from torch.autograd import Variable


# Model definition
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        def building_block(indim, outdim):
            return [
                nn.Conv2d(indim, outdim, 3, 1, 1),
                nn.BatchNorm2d(outdim),
                nn.ReLU(),
            ]

        def stack_blocks(indim, outdim, block_num):
            layers = building_block(indim, outdim)
            for i in range(block_num - 1):
                layers += building_block(outdim, outdim)
            layers.append(nn.MaxPool2d(2, 2, 0))
            return layers

        cnn_list = []
        cnn_list += stack_blocks(3, 128, 3)
        cnn_list += stack_blocks(128, 128, 3)
        cnn_list += stack_blocks(128, 256, 3)
        cnn_list += stack_blocks(256, 512, 1)
        cnn_list += stack_blocks(512, 512, 1)
        self.cnn = nn.Sequential( * cnn_list)

        dnn_list = [
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(1024, 11),
        ]
        self.fc = nn.Sequential( * dnn_list)

    def forward(self, x):
        out = self.cnn(x)
        out = out.reshape(out.size()[0], -1)
        return self.fc(out)

def normalize(image):
    return (image - image.min()) / (image.max() - image.min())
    # return torch.log(image)/torch.log(image.max())

def compute_saliency_maps(x, y, model):
    """ Saliency map explanation. """
    model.eval()
    x = x.cuda()

    # we want the gradient of the input x
    x.requires_grad_()

    y_pred = model(x)
    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(y_pred, y.cuda())
    loss.backward()

    # saliencies = x.grad.abs().detach().cpu()
    saliencies, _ = torch.max(x.grad.data.abs().detach().cpu(),dim=1)

    # We need to normalize each image, because their gradients might vary in scale
    saliencies = torch.stack([normalize(item) for item in saliencies])
    return saliencies

def smooth_grad(x, y, model, epoch, param_sigma_multiplier):
    model.eval()
    #x = x.cuda().unsqueeze(0)

    mean = 0
    sigma = param_sigma_multiplier / (torch.max(x) - torch.min(x)).item()
    smooth = np.zeros(x.cuda().unsqueeze(0).size())

    for i in range(epoch):
        # call Variable to generate random noise
        noise = Variable(x.data.new(x.size()).normal_(mean, sigma**2))
        x_mod = (x + noise).unsqueeze(0).cuda()
        x_mod.requires_grad_()

        y_pred = model(x_mod)
        loss_func = torch.nn.CrossEntropyLoss()
        loss = loss_func(y_pred, y.cuda().unsqueeze(0))
        loss.backward()

        # like the method in saliency map
        smooth += x_mod.grad.abs().detach().cpu().data.numpy()

    smooth = normalize(smooth / epoch) # don't forget to normalize
    # smooth = smooth / epoch # try this line to answer the question

    return smooth

class IntegratedGradients:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()

    def generate_images_on_linear_path(self, input_image, steps):
        # Generate scaled xbar images
        xbar_list = [input_image*step/steps for step in range(steps)]
        return xbar_list

    def generate_gradients(self, input_image, target_class):
        # We want to get the gradients of the input image
        input_image.requires_grad=True
        # Forward
        model_output = self.model(input_image)
        # Zero grads
        self.model.zero_grad()
        # Target for back propagation
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_().cuda()
        one_hot_output[0][target_class] = 1
        # Backward
        model_output.backward(gradient=one_hot_output)
        self.gradients = input_image.grad
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,128,128)
        gradients_as_arr = self.gradients.data.cpu().numpy()[0]
        return gradients_as_arr

    def generate_integrated_gradients(self, input_image, target_class, steps):
        # Generate xbar images
        xbar_list = self.generate_images_on_linear_path(input_image, steps)

        # Initialize an image composed of zeros
        integrated_grads = np.zeros(input_image.size())
        for xbar_image in xbar_list:
            # Generate gradients from xbar images
            single_integrated_grad = self.generate_gradients(xbar_image, target_class)
            # Add rescaled grads from xbar images
            integrated_grads = integrated_grads + single_integrated_grad/steps

        # [0] to get rid of the first channel (1,3,128,128)
        return integrated_grads[0]