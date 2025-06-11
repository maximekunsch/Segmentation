import numpy as np
from skimage.morphology import white_tophat, thin, disk, star, square, diamond, octagon, rectangle, convex_hull_image
from skimage.filters.rank import entropy
from PIL import Image
import cv2
from matplotlib import pyplot as plt

def my_segmentation(img, img_mask):
    # Inversion of colors and mask
    img_pretreat = (255 * img_mask & (np.invert(img))).astype(np.uint8)

    # Multi-scale top-hat transformation
    scales = [3, 5, 7]  # Different scales for the top-hat
    # Initialization of the output image for multi-scale top-hat
    img_tophat_multi = np.zeros_like(img_pretreat)

    # Applying white top-hat for each scale and combining the results
    for scale in scales:
        img_tophat_multi = np.maximum(img_tophat_multi, white_tophat(img_pretreat, disk(scale)))

    # Intensity threshold
    threshold = 15
    img_binary = img_tophat_multi > threshold

    # Conversion for connected component processing
    # Conversion of the binary image to uint8 for cv2
    threshold_uint8 = (img_binary * 255).astype(np.uint8)

    # Find connected components of the binary image and filter small components
    # Use cv2 for connected components
    num_comp, comps = cv2.connectedComponents(threshold_uint8)

    # Filter components by size
    img_connexe = np.zeros_like(img_binary)
    for comp in range(1, num_comp):
        component = (comps == comp)
        component_size = np.sum(component)
        if component_size > 25:  # Minimum size for capillaries
            img_connexe |= component

    # Circular mask to focus on the region of interest
    mask = np.zeros_like(img_connexe, dtype=np.uint8)
    mask = cv2.circle(mask, (256, 256), 253, 1, -1)

    # Applying the circular mask to keep only the pixels within the inscribed disk
    img_out = img_connexe & mask

    return img_out

def evaluate(img_out, img_GT):
    GT_skel = thin(img_GT, max_num_iter=15)
    img_out_skel = thin(img_out, max_num_iter=15)
    TP = np.sum(img_out_skel & img_GT)
    FP = np.sum(img_out_skel & ~img_GT)
    FN = np.sum(GT_skel & ~img_out)

    ACCU = TP / (TP + FP)
    RECALL = TP / (TP + FN)
    return ACCU, RECALL, img_out_skel, GT_skel

# Open the original image in grayscale
img = np.asarray(Image.open('./images_IOSTAR/star02_OSC.jpg')).astype(np.uint8)
print(img.shape)

nrows, ncols = img.shape
row, col = np.ogrid[:nrows, :ncols]

# Consider only the pixels within the inscribed disk
img_mask = (np.ones(img.shape)).astype(np.bool_)
invalid_pixels = ((row - nrows/2)**2 + (col - ncols/2)**2 > (nrows / 2)**2)
img_mask[invalid_pixels] = 0

img_out = my_segmentation(img, img_mask)

# Open the Ground Truth image as boolean
img_GT = np.asarray(Image.open('./images_IOSTAR/GT_02.png')).astype(np.uint8)

ACCU, RECALL, img_out_skel, GT_skel = evaluate(img_out, img_GT)
print('Accuracy =', ACCU, ', Recall =', RECALL)

plt.subplot(231)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.subplot(232)
plt.imshow(img_out, cmap='gray')
plt.title('Segmentation')
plt.subplot(233)
plt.imshow(img_out_skel, cmap='gray')
plt.title('Segmentation Skeleton')
plt.subplot(235)
plt.imshow(img_GT, cmap='gray')
plt.title('Ground Truth')
plt.subplot(236)
plt.imshow(GT_skel, cmap='gray')
plt.title('Ground Truth Skeleton')
plt.show()
