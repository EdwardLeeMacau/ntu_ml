import os

from config import initialize
from matplotlib import pyplot as plt
from model import normalize
from torch.optim import Adam
from tqdm import tqdm

args, model, img_indices, images, labels = initialize()

layer_activations = None
def filter_explanation(x, model, cnn_id, filter_id, iteration=100, lr=1):
    # x: input image
    # cnn_id: cnn layer id
    # filter_id: which filter
    model.eval()

    def hook(model, input, output):
        global layer_activations
        layer_activations = output

    hook_handle = model.cnn[cnn_id].register_forward_hook(hook)
    # When the model forwards through the layer[cnn_id], it needs to call the hook function first
    # The hook function save the output of the layer[cnn_id]
    # After forwarding, we'll have the loss and the layer activation

    # Filter activation: x passing the filter will generate the activation map
    model(x.cuda()) # forward

    # Based on the filter_id given by the function argument, pick up the specific filter's activation map
    # We just need to plot it, so we can detach from graph and save as cpu tensor
    filter_activations = layer_activations[:, filter_id, :, :].detach().cpu()

    # Filter visualization: find the image that can activate the filter the most
    x = x.cuda()
    x.requires_grad_()
    # input image gradient
    optimizer = Adam([x], lr=lr)
    # Use optimizer to modify the input image to amplify filter activation
    for iter in tqdm(range(iteration), ncols=0, desc="CNN Layer {}".format(cnn_id)):
        optimizer.zero_grad()
        model(x)

        objective = -layer_activations[:, filter_id, :, :].sum()
        # We want to maximize the filter activation's summation
        # So we add a negative sign

        objective.backward()
        # Calculate the partial differential value of filter activation to input image
        optimizer.step()
        # Modify input image to maximize filter activation
    filter_visualizations = x.detach().cpu().squeeze()

    # Don't forget to remove the hook
    hook_handle.remove()
    # The hook will exist after the model register it, so you have to remove it after used
    # Just register a new hook if you want to use it

    return filter_activations, filter_visualizations

def visualize_filter(cnn_id: int, filter_id: int):
    filter_activations, filter_visualizations = filter_explanation(
        images, model, cnn_id=cnn_id, filter_id=filter_id, iteration=100, lr=0.1
    )

    fig, axs = plt.subplots(3, len(img_indices), figsize=(15, 8))

    # Plot raw image
    for i, img in enumerate(images):
        axs[0][i].imshow(img.permute(1, 2, 0))

    # Plot filter activations
    for i, img in enumerate(filter_activations):
        axs[1][i].imshow(normalize(img))

    # Plot filter visualization
    for i, img in enumerate(filter_visualizations):
        axs[2][i].imshow(normalize(img.permute(1, 2, 0)))


    os.makedirs('.explain', exist_ok=True)
    plt.savefig(f'.explain/cnn{cnn_id}_filter{filter_id}.png')
    plt.close()

visualize_filter(cnn_id=6, filter_id=0)
visualize_filter(cnn_id=15, filter_id=0)
visualize_filter(cnn_id=23, filter_id=0)
