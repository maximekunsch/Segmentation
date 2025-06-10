import numpy as np
from skimage.morphology import white_tophat, thin, disk, star, square, diamond, octagon, rectangle, convex_hull_image
from skimage.filters.rank import entropy
from PIL import Image
import cv2
from matplotlib import pyplot as plt

def my_segmentation(img, img_mask):
    # Inversion des couleurs et mask
    img_pretreat = (255 * img_mask & (np.invert(img))).astype(np.uint8)

    # Multi-scales top-hat transformation
    scales = [3,5,7]  # Différentes échelles pour le top-hat
    # Initialisation de l'image de sortie
    # pour le top-hat multi-échelle
    img_tophat_multi = np.zeros_like(img_pretreat)

    # Application du top-hat blanc pour chaque échelle
    # et combinaison des résultats
    for scale in scales:
        img_tophat_multi = np.maximum(img_tophat_multi, white_tophat(img_pretreat, disk(scale)))

    # Seuil de l'intensité
    seuil = 15
    img_binaire = img_tophat_multi > seuil

    # Conversion pour le traitement des composantes connectées
    # Conversion de l'image binaire en uint8 pour cv2
    seuil_uint8 = (img_binaire * 255).astype(np.uint8)

    # Trouver les composantes connectées de l'image binaire
    # et filtrer les petites composantes
    # Utilisation de cv2 pour les composantes connectées
    num_comp, comps = cv2.connectedComponents(seuil_uint8)

    # Filtrer les composantes par taille
    img_connexe = np.zeros_like(img_binaire)
    for comp in range(1, num_comp):
        component = (comps == comp)
        component_size = np.sum(component)
        if component_size > 25:  # Taille minimale pour les capillaires
            img_connexe |= component

    # Masque circulaire pour se concentrer sur la région d'intérêt
    mask = np.zeros_like(img_connexe, dtype=np.uint8)
    mask = cv2.circle(mask, (256, 256), 253, 1, -1)

    # Application du masque circulaire
    # pour ne garder que les pixels dans le disque inscrit
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

#Ouvrir l'image originale en niveau de gris
img =  np.asarray(Image.open('./images_IOSTAR/star02_OSC.jpg')).astype(np.uint8)
print(img.shape)

nrows, ncols = img.shape
row, col = np.ogrid[:nrows, :ncols]
#On ne considere que les pixels dans le disque inscrit 
img_mask = (np.ones(img.shape)).astype(np.bool_)
invalid_pixels = ((row - nrows/2)**2 + (col - ncols/2)**2 > (nrows / 2)**2)
img_mask[invalid_pixels] = 0

img_out = my_segmentation(img,img_mask)

#Ouvrir l'image Verite Terrain en booleen
img_GT =  np.asarray(Image.open('./images_IOSTAR/GT_02.png')).astype(np.uint8)

ACCU, RECALL, img_out_skel, GT_skel = evaluate(img_out, img_GT)
print('Accuracy =', ACCU,', Recall =', RECALL)

plt.subplot(231)
plt.imshow(img,cmap = 'gray')
plt.title('Image Originale')
plt.subplot(232)
plt.imshow(img_out,cmap='gray')
plt.title('Segmentation')
plt.subplot(233)
plt.imshow(img_out_skel,cmap='gray')
plt.title('Segmentation squelette')
plt.subplot(235)
plt.imshow(img_GT,cmap='gray')
plt.title('Verite Terrain')
plt.subplot(236)
plt.imshow(GT_skel,cmap='gray')
plt.title('Verite Terrain Squelette')
plt.show()
