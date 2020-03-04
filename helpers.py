import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.misc import imresize
from matplotlib.patches import Circle


def show_images(imgs, per_row=3, per_col=2, W=10, H=5, tdpi=80):
    _, ax = plt.subplots(per_col, per_row, figsize=(W, H), dpi=tdpi)
    ax = ax.ravel()

    for i in range(len(imgs)):
        img = imgs[i]
        ax[i].imshow(img)

    for i in range(per_row * per_col):
        ax[i].axis('off')


def show_dotted_image(image, points, thickness=5, color=[255, 0, 255], dim=15):
    copied_image = image.copy()

    cv2.line(copied_image, points[0], points[1], color, thickness)
    cv2.line(copied_image, points[2], points[3], color, thickness)

    _, ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(copied_image)

    for (x, y) in points:
        dot = Circle((x, y), dim)
        ax.add_patch(dot)


def roi(image, right=125, left=1200):
    m = np.copy(image) + 1
    m[:, :right] = 0
    m[:, left:] = 0

    return m


def scale_abs(x, m=255):
    x = np.absolute(x)
    x = np.uint8(m * x / np.max(x))

    return x


def color_heat_map(heatMapMono, cmap=cv2.COLORMAP_HOT):
    heatMapInt = cv2.equalizeHist(heatMapMono.astype(np.uint8))
    heatColor = cv2.applyColorMap(heatMapInt, cmap)
    heatColor = cv2.cvtColor(heatColor, code=cv2.COLOR_BGR2RGB)

    return heatColor


def draw_boxes(image, boxes, color=(0, 255, 0), thickness=4):
    for box in boxes:

        box = np.array(box)
        box = box.reshape(box.size)

        cv2.rectangle(img=image, pt1=(box[0], box[1]), pt2=(
            box[2], box[3]), color=color, thickness=thickness)

    return image


def flip_image(image):
    return cv2.flip(image, 1)


def add_small_image(pip_image, original_image, corner, pip_alpha = 0.5):

    pip_image = imresize(pip_image, size=0.3)

    pip_height = pip_image.shape[0]
    pip_width = pip_image.shape[1]

    original_width = original_image.shape[1]

    if corner == 'TOP-LEFT':
        origin = (20, 20)
    if corner == 'TOP-RIGHT':
        origin = (original_width - pip_width - 20, 20)

    background = original_image[origin[1]:origin[1] + pip_height, origin[0]:origin[0] + pip_width]

    blend = np.round(background * (1 - pip_alpha), 0) + np.round(pip_image, 0)
    blend = np.minimum(blend, 255)

    original_image[origin[1]:origin[1] + pip_height, origin[0]:origin[0] + pip_width] = blend

    return original_image    

    
    
