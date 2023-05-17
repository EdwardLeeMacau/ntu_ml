import matplotlib.pyplot as plt

# def no_axis_show(img, title='', cmap=None):
#   # imshow, and set the interpolation mode to be "nearest"。
#   fig = plt.imshow(img, interpolation='nearest', cmap=cmap)
#   # do not show the axes in the images.
#   fig.axes.get_xaxis().set_visible(False)
#   fig.axes.get_yaxis().set_visible(False)
#   plt.title(title)

# titles = ['horse', 'bed', 'clock', 'apple', 'cat', 'plane', 'television', 'dog', 'dolphin', 'spider']
# plt.figure(figsize=(18, 18))
# for i in range(10):
#   plt.subplot(1, 10, i+1)
#   fig = no_axis_show(plt.imread(f'real_or_drawing/train_data/{i}/{500*i}.bmp'), title=titles[i])

# plt.figure(figsize=(18, 18))
# for i in range(10):
#   plt.subplot(1, 10, i+1)
#   fig = no_axis_show(plt.imread(f'real_or_drawing/test_data/0/' + str(i).rjust(5, '0') + '.bmp'))

# """# Special Domain Knowledge

# When we graffiti, we usually draw the outline only, therefore we can perform edge detection processing on the source data to make it more similar to the target data.


# ## Canny Edge Detection
# The implementation of Canny Edge Detection is as follow.
# The algorithm will not be describe thoroughly here.  If you are interested, please refer to the wiki or [here](https://medium.com/@pomelyu5199/canny-edge-detector-%E5%AF%A6%E4%BD%9C-opencv-f7d1a0a57d19).

# We only need two parameters to implement Canny Edge Detection with CV2:  `low_threshold` and `high_threshold`.

# ```cv2.Canny(image, low_threshold, high_threshold)```

# Simply put, when the edge value exceeds the high_threshold, we determine it as an edge. If the edge value is only above low_threshold, we will then determine whether it is an edge or not.

# Let's implement it on the source data.
# """

import cv2
# import matplotlib.pyplot as plt
# titles = ['horse', 'bed', 'clock', 'apple', 'cat', 'plane', 'television', 'dog', 'dolphin', 'spider']
# plt.figure(figsize=(18, 18))

# original_img = plt.imread(f'real_or_drawing/train_data/0/0.bmp')
# plt.subplot(1, 5, 1)
# no_axis_show(original_img, title='original')

# gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
# plt.subplot(1, 5, 2)
# no_axis_show(gray_img, title='gray scale', cmap='gray')

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