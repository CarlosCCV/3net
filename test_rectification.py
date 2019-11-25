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

    left = tf.image.convert_image_dtype(left, tf.float32)
    left = tf.image.resize_images(left, [256, 512], tf.image.ResizeMethod.AREA)

    center = tf.image.convert_image_dtype(center, tf.float32)
    center = tf.image.resize_images(center, [256, 512], tf.image.ResizeMethod.AREA)

    right = tf.image.convert_image_dtype(right, tf.float32)
    right = tf.image.resize_images(right, [256, 512], tf.image.ResizeMethod.AREA)

    # offset = tf.constant(5, dtype=tf.float32)
    # center = center + offset

    # # Rectification using TensorFlow
    # left_rect, cl_rect, _, _ = stereo_rectification.stereo_rectify(left, center, intrinsics[:, :, 0], intrinsics[:, :, 1], extrinsics[:, :, 0], extrinsics[:, :, 1], transformed_image_size=(256, 512))
    # cr_rect, right_rect, _, _ = stereo_rectification.stereo_rectify(center, right, intrinsics[:, :, 1], intrinsics[:, :, 2], extrinsics[:, :, 1], extrinsics[:, :, 2], transformed_image_size=(256, 512))
    #
    # maximum_cr = tf.reduce_max(cr_rect)
    #
    # # Rectification using OpenCV
    # center_cv = center_cv + 5
    # left_rect_cv, cl_rect_cv, _, _ = stereo_rectification.stereo_rectify(left_cv, center_cv, intrinsics[:, :, 0], intrinsics[:, :, 1], extrinsics[:, :, 0], extrinsics[:, :, 1], False, transformed_image_size=(256, 512))
    # cr_rect_cv, right_rect_cv, _, _ = stereo_rectification.stereo_rectify(center_cv, right_cv, intrinsics[:, :, 1], intrinsics[:, :, 2], extrinsics[:, :, 1], extrinsics[:, :, 2], False,transformed_image_size=(256, 512))
    #
    # # Unrectification with TensorFlow
    # left_unrect, cl_unrect = stereo_rectification.unrectify(left_rect, cl_rect, intrinsics[:,:,0], intrinsics[:,:,1], extrinsics[:,:,0], extrinsics[:,:,1], transformed_image_size=(256, 512))
    # cr_unrect, right_unrect = stereo_rectification.unrectify(cr_rect, right_rect, intrinsics[:, :, 1], intrinsics[:, :, 2], extrinsics[:, :, 1], extrinsics[:, :, 2], transformed_image_size=(256, 512))
    #
    # # Unrectification with OpenCV
    # left_unrect_cv, cl_unrect_cv = stereo_rectification.unrectify(left_rect_cv, cl_rect_cv, intrinsics[:, :, 0],
    #                                                         intrinsics[:, :, 1], extrinsics[:, :, 0],
    #                                                         extrinsics[:, :, 1], False,
    #                                                         transformed_image_size=(256, 512))
    # cr_unrect_cv, right_unrect_cv = stereo_rectification.unrectify(cr_rect_cv, right_rect_cv, intrinsics[:, :, 1],
    #                                                          intrinsics[:, :, 2], extrinsics[:, :, 1],
    #                                                          extrinsics[:, :, 2], False,
    #                                                          transformed_image_size=(256, 512))
    #
    # mask_cl = create_mask(cl_rect)
    # mask_cr = create_mask(cr_rect)
    # mask_cl = tf.cast(mask_cl, dtype=tf.float32)
    # mask_cr = tf.cast(mask_cr, dtype=tf.float32)
    #
    # cl_image = cl_rect - offset
    # cr_image = cr_rect - offset
    #
    # cl_image = tf.multiply(cl_image, mask_cl)
    # cr_image = tf.multiply(cr_image, mask_cr)
    #
    #
    # sum_cl = tf.reduce_sum(cl_unrect)
    # sum_cr = tf.reduce_sum(cr_unrect)
    #
    # sum_cl_cv = np.sum(cl_unrect_cv)
    # sum_cr_cv = np.sum(cr_unrect_cv)
    #
    # result = tf.cond(sum_cl < sum_cr, lambda: create_mask(cl_unrect), lambda: create_mask(cr_unrect))
    # result = tf.cast(result, dtype = tf.float32)
    #
    # if sum_cl_cv < sum_cr_cv:
    #     mask_cv = create_mask(cl_unrect_cv)
    #     print('CL used')
    # else:
    #     mask_cv = create_mask(cr_unrect_cv)
    #     print('CR used')
    #
    # cl_unrect = cl_unrect - 5
    # cr_unrect = cr_unrect - 5
    # cl_masked = tf.multiply(cl_unrect, result)
    # cr_masked = tf.multiply(cr_unrect, result)

    # Rectification using OpenCV
    # center_cv = center_cv + 5
    print(center_cv[82, 507, 0])
    print(center_cv[82, 507, 1])
    print(center_cv[82,507,2])
    print('---------------------------------------------------')
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
    left_unrect, cl_unrect = stereo_rectification.unrectify(left_rect, cl_rect, intrinsics[:, :, 0],
                                                            intrinsics[:, :, 1], extrinsics[:, :, 0],
                                                            extrinsics[:, :, 1], False,
                                                            transformed_image_size=(256, 512))
    cr_unrect, right_unrect = stereo_rectification.unrectify(cr_rect, right_rect, intrinsics[:, :, 1],
                                                             intrinsics[:, :, 2], extrinsics[:, :, 1],
                                                             extrinsics[:, :, 2], False,
                                                             transformed_image_size=(256, 512))
    print(center_cv[82, 507, 0])
    print(center_cv[82, 507, 1])
    print(center_cv[82,507,2])
    print('---------------------------------------------------')
    sum_cl = np.sum(cl_unrect)
    sum_cr = np.sum(cr_unrect)


    if (sum_cl < sum_cr):
        print('CL')
        mask_cv = create_mask(cl_unrect)
    else:
        print('CR')
        mask_cv = create_mask(cr_unrect)

    print(cr_unrect[82, 507, 0])
    print(cr_unrect[82, 507, 1])
    print(cr_unrect[82,507,2])
    print('---------------------------------------------------')
    print(mask_cv.astype(float)[0,0,0])
    print(mask_cv.astype(float)[82,507, 1])
    print(mask_cv.astype(float)[82, 507, 2])
# # SESSION
# config = tf.ConfigProto(allow_soft_placement=True)
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
#
# # INIT
# sess.run(tf.global_variables_initializer())
# sess.run(tf.local_variables_initializer())
# coordinator = tf.train.Coordinator()
#
# left_im, center_im, right_im, left_rectified, cl_rectified, cr_rectified, right_rectified, left_unrectified, cl_unrectified, cr_unrectified, right_unrectified, suma_cl, suma_cr, maskara, cl_mask, cr_mask, cl, cr, maxi = \
#     sess.run([left, center, right, left_rect, cl_rect, cr_rect, right_rect, left_unrect, cl_unrect, cr_unrect, right_unrect, sum_cl, sum_cr, mask_cr, cl_masked, cr_masked, cl_image, cr_image, maximum_cr])
#
#
# print(suma_cl)
# print(suma_cr)
#
# for i in range(0, left_im.shape[0], 50):
#     cv2.line(left_im, (0, i), (left_im.shape[1], i), (255, 0, 0), 2)
#     cv2.line(center_im, (0, i), (left_im.shape[1], i), (255, 0, 0), 2)
#     cv2.line(left_rectified, (0, i), (left_rectified.shape[1], i), (255, 0, 0), 2)
#     cv2.line(cl_rectified, (0, i), (cl_rectified.shape[1], i), (255, 0, 0), 2)
#     cv2.line(right_rectified, (0, i), (right_rectified.shape[1], i), (255, 0, 0), 2)
#     cv2.line(cr_rectified, (0, i), (cr_rectified.shape[1], i), (255, 0, 0), 2)


# plt.figure(0, figsize=(12,10))
#
# plt.tight_layout()
#
#
# plt.figure(1, figsize=(12,10))
# plt.subplot(221)
# plt.title('center original')
# plt.imshow(center_im, cmap='gray')
# plt.subplot(222)
# plt.title('right original')
# plt.imshow(right_im, cmap='gray')
# plt.subplot(223)
# plt.title('cr rectified')
# plt.imshow(cr_rectified, cmap='gray')
# plt.subplot(224)
# plt.title('right rectified')
# plt.imshow(right_rectified, cmap='gray')
# plt.tight_layout()
# plt.show()
#
# plt.figure(2, figsize=(12,10))
# plt.subplot(221)
# plt.title('left original')
# plt.imshow(left_im, cmap='gray')
# plt.subplot(222)
# plt.title('center original')
# plt.imshow(center_im, cmap='gray')
# plt.subplot(223)
# plt.title('left unrectified')
# plt.imshow(left_unrectified, cmap='gray')
# plt.subplot(224)
# plt.title('cl unrectified')
# plt.imshow(cl_unrectified, cmap='gray')
# plt.tight_layout()
# plt.show()
#
# plt.figure(3, figsize=(12,10))
# plt.subplot(221)
# plt.title('center original')
# plt.imshow(left_im, cmap='gray')
# plt.subplot(222)
# plt.title('right original')
# plt.imshow(center_im, cmap='gray')
# plt.subplot(223)
# plt.title('cr unrectified')
# plt.imshow(cr_unrectified, cmap='gray')
# plt.subplot(224)
# plt.title('right unrectified')
# plt.imshow(right_unrectified, cmap='gray')
# plt.tight_layout()
# plt.show()

# print(cl)
# print('---------------------------------------------------------')
# print(np.max(cl))
# print(maxi)
# print('---------------------------------------------------------')
# print(maskara)
#
# cl = cl - 0.5
plt.figure(4, figsize=(12,10))
cv2.imwrite('cr_rect.png', cr_rect)
cv2.imwrite('cl_rect.png', cl_rect)
cv2.imwrite('cr_unrect.png', cr_unrect)
cv2.imwrite('cl_unrect.png', cl_unrect)
cv2.imwrite('mask.png', mask_cv*255)
cv2.imwrite('mask_cr.png', mask_cr*255)
cv2.imwrite('mask_cl.png', mask_cl*255)
# plt.imshow((mask_cv.astype(float) > 0).astype(float), cmap='gray')
plt.imshow(mask_cv*255)
plt.show()

print(filename)

