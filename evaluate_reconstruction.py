import numpy as np
import os
from matplotlib import pyplot as plt
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from sklearn.metrics import mean_squared_error as mse

directory = "/media/ccv/Maxtor/DATASETS/FVV/OUTPUTS/MONODEPTH/SIMPLE_SEQUENCES"

original = np.load(os.path.join(directory,'img_right_original.npy'))
reconstructed = np.load(os.path.join(directory,'img_right_reconstructed.npy'))

original = (original*255).astype(int)
reconstructed = (reconstructed*255).astype(int)

for j in range(5):
    plt.figure(j)
    f, axarr = plt.subplots(4 // 2, 2)
    for i in range(2):
        axarr[0, i].imshow(original[j*2+i,:,:])
        axarr[0, i].set_title('ORIGINAL')
        axarr[1, i].imshow(reconstructed[j*2+i, :, :, :])
        axarr[1, i].set_title('RECONSTRUCTED')

plt.show()


# METRICS
ssim_array = np.zeros(original.shape[0], dtype = np.float32)
psnr_array = np.zeros(original.shape[0], dtype = np.float32)
mse_per_channel = np.zeros((original.shape[0],3), dtype = np.float32)

for i in range(original.shape[0]):
    ssim_array[i] = ssim(original[i, :, :, :], reconstructed[i, :, :, :], multichannel=True, data_range=255)
    psnr_array[i] = psnr(original[i,:,:,:], reconstructed[i, :, :, :], data_range=255)
    for j in range(3):
        mse_per_channel[i,j] = mse(original[i,:,:,j], reconstructed[i,:,:,j])

ssim_array_mean = np.mean(ssim_array)
psnr_array_mean = np.mean(psnr_array)

#MSE
mse_array = np.mean(mse_per_channel, axis=1)
mse_result = np.mean(mse_array)

print("The Mean SSIM is {}".format(ssim_array_mean))
print("The MSE is {}".format(mse_result))
print("The Mean PSNR is {}".format(psnr_array_mean))
