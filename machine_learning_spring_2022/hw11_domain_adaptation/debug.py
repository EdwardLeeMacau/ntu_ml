import cv2
import matplotlib.pyplot as plt
import yaml
from utils import same_seeds

# load hyperparameters from yaml
with open("params.yaml", "r") as f:
    hparams = yaml.load(f, Loader=yaml.FullLoader)

# fix random seed for reproducibility
same_seeds(hparams["seed"])
root_dir = "/tmp2/edwardlee/dataset/real_or_drawing/"

titles = [
    'horse', 'bed', 'clock', 'apple', 'cat', 'plane', 'television', 'dog', 'dolphin', 'spider'
]

def no_axis_show(img, title='', cmap=None):
    # imshow, and set the interpolation mode to be "nearest"。
    fig = plt.imshow(img, interpolation='nearest', cmap=cmap)
    # do not show the axes in the images.
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.title(title)

# Show canny edge detection as preprocessing
# plt.figure(figsize=(18, 18))

# original_img = plt.imread(f'real_or_drawing/train_data/0/0.bmp')
# plt.subplot(1, 5, 1)
# no_axis_show(original_img, title='original')

# gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
# plt.subplot(1, 5, 2)
# no_axis_show(gray_img, title='gray scale', cmap='gray')

# canny_50100 = cv2.Canny(gray_img, 50, 100)
# plt.subplot(1, 5, 3)
# no_axis_show(canny_50100, title='Canny(50, 100)', cmap='gray')

# canny_150200 = cv2.Canny(gray_img, 150, 200)
# plt.subplot(1, 5, 4)
# no_axis_show(canny_150200, title='Canny(150, 200)', cmap='gray')

# canny_250300 = cv2.Canny(gray_img, 250, 300)
# plt.subplot(1, 5, 5)
# no_axis_show(canny_250300, title='Canny(250, 300)', cmap='gray')

