from read_parameters import read_calibration_parameters
import tensorflow as tf
import stereo_rectification
from matplotlib import pyplot as plt
import cv2
import numpy as np

def create_mask(image):
    mask = np.zeros((256,512,3), dtype = float)
    for i in range(256):
        for j in range(512):
            mask[i,j,:] = (~(image[i,j,0] == 101 and image[i,j,1] == 101 and image[i,j,2] == 101)).astype(float)
    return mask


filenames_file = '/mnt/goatse/DATASETS/FVV/filenames/train_trinocular_val_3.txt'

intrinsics, extrinsics = read_calibration_parameters('', filenames_file)

file = open(filenames_file, 'r')

contents = file.readlines()

filename = contents[0].split()

left_file = filename[0]
center_file = filename[1]
right_file = filename[2]
print(left_file)
print(center_file)
print(right_file)

with tf.device('/gpu:0'):
    # Define graph
    left = tf.image.decode_png(tf.read_file(left_file))
    center = tf.image.decode_png(tf.read_file(center_file))
    right = tf.image.decode_png(tf.read_file(right_file))

    left_cv = cv2.imread(left_file)
    center_cv = cv2.imread(center_file)
    right_cv = cv2.imread(right_file)

    left_cv = cv2.resize(left_cv, (512,256), interpolation=cv2.INTER_AREA)
    center_cv = cv2.resize(center_cv, (512,256), interpolation=cv2.INTER_AREA)
    right_cv = cv2.resize(right_cv, (512, 256), interpolation=cv2.INTER_AREA)

    left = tf.image.convert_image_dtype(left, tf.float32)
    left = tf.image.resize_images(left, [256, 512], tf.image.ResizeMethod.AREA)

    center = tf.image.convert_image_dtype(center, tf.float32)
    center = tf.image.resize_images(center, [256, 512], tf.image.ResizeMethod.AREA)

    right = tf.image.convert_image_dtype(right, tf.float32)
    right = tf.image.resize_images(right, [256, 512], tf.image.ResizeMethod.AREA)


    # Rectification using OpenCV
    left_rect, cl_rect = stereo_rectification.stereo_rectify(left_cv, center_cv, intrinsics[:, :, 0],
                                                                   intrinsics[:, :, 1], extrinsics[:, :, 0],
                                                                   extrinsics[:, :, 1], False,
                                                                   transformed_image_size=(256, 512))
    cr_rect, right_rect = stereo_rectification.stereo_rectify(center_cv, right_cv, intrinsics[:, :, 1],
                                                                    intrinsics[:, :, 2], extrinsics[:, :, 1],
                                                                    extrinsics[:, :, 2], False,
                                                                    transformed_image_size=(
                                                                        256, 512))
    mask_cr = create_mask(cr_rect)
    mask_cl = create_mask(cl_rect)

    # Unrectification using OpenCV
    left_unrect, cl_unrect = stereo_rectification.unrectify(left_rect, cl_rect, intrinsics[:, :, 0],
                                                            intrinsics[:, :, 1], extrinsics[:, :, 0],
                                                            extrinsics[:, :, 1], False,
                                                            transformed_image_size=(256, 512))
    cr_unrect, right_unrect = stereo_rectification.unrectify(cr_rect, right_rect, intrinsics[:, :, 1],
                                                             intrinsics[:, :, 2], extrinsics[:, :, 1],
                                                             extrinsics[:, :, 2], False,
                                                             transformed_image_size=(256, 512))

    # Create mask from the unrectified image where more black area is got
    sum_cl = np.sum(cl_unrect)
    sum_cr = np.sum(cr_unrect)

    if (sum_cl < sum_cr):
        print('CL')
        mask_cv = create_mask(cl_unrect)
    else:
        print('CR')
        mask_cv = create_mask(cr_unrect)

for i in range(20,left_rect.shape[0],20):
    cv2.line(right_rect, (0,i), (511, i), (255,0,0))
    cv2.line(cr_rect, (0,i), (511, i), (255,0,0))



plt.figure(0, figsize=(12,10))
plt.subplot(2,2,1)
plt.imshow(center_cv)
plt.subplot(2,2,2)
plt.imshow(right_cv)
plt.subplot(2,2,3)
plt.imshow(cr_rect)
plt.subplot(2,2,4)
plt.imshow(right_rect)

plt.show()


cv2.imwrite('cr_rect.png', cr_rect)
cv2.imwrite('cl_rect.png', cl_rect)
cv2.imwrite('cr_unrect.png', cr_unrect)
cv2.imwrite('cl_unrect.png', cl_unrect)
cv2.imwrite('mask.png', mask_cv*255)
cv2.imwrite('mask_cr.png', mask_cr*255)
cv2.imwrite('mask_cl.png', mask_cl*255)


