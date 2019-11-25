import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

testing_dir = "/mnt/goatse/DATASETS/FVV/Models/fvv_trinet_save_mask"
output_dir = os.path.join(testing_dir,'PNG/')




disp_cl = np.load(os.path.join(testing_dir,'disparity_cl.npy'))
disp_cr = np.load(os.path.join(testing_dir,'disparity_cr.npy'))
cl = np.load(os.path.join(testing_dir,'cl_unrect.npy'))
cr = np.load(os.path.join(testing_dir,'cr_unrect.npy'))
cl_original = np.load(os.path.join(testing_dir,'cl_original.npy'))
cr_original = np.load(os.path.join(testing_dir,'cr_original.npy'))

if not os.path.exists(output_dir):
    os.mkdir(output_dir)


for i in range(10):
    disparity_cl = disp_cl[i, ...]
    disparity_cr = disp_cr[i, ...]
    cl_rec = cl[i, ...]
    cr_rec = cr[i, ...]

    disparity_cl = disparity_cl * 255 / np.amax(disparity_cl)
    disparity_cr = disparity_cr * 255 / np.amax(disparity_cr)
    cv2.imwrite(output_dir + 'disp_cl_{}.png'.format(i), disparity_cl)


    plt.figure(0, figsize=(12,10))
    plt.subplot(231)
    plt.title('cl original')
    plt.imshow(cl_original[i,...])
    plt.subplot(232)
    plt.title('cl reconstruction')
    plt.imshow(cl_rec, cmap='gray')
    plt.subplot(233)
    plt.title('cl depth')
    plt.imshow(disparity_cl, cmap='gray')
    plt.subplot(234)
    plt.title('cr original')
    plt.imshow(cr_original[i,...], cmap='gray')
    plt.subplot(235)
    plt.title('cr reconstruction')
    plt.imshow(cr_rec, cmap='gray')
    plt.subplot(236)
    plt.title('cr depth estimation')
    plt.imshow(disparity_cr, cmap='gray')
    plt.tight_layout()
    plt.show()