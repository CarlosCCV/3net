import numpy as np
import cv2
import tensorflow as tf
import os

def stereo_rectify(image1, image2, A1, A2, ext1, ext2, TF = True, transformed_image_size = (1080,1920), calibration_image_size = (1080,1920), directory = None):

    R1 = ext1[0:3,0:3]

    # We need to adapt the intrinsics matrix to the change in size we want to get
    multiplier = np.divide(transformed_image_size, calibration_image_size)

    A1_resized = np.copy(A1)
    A1_resized[0, 0] = A1[0, 0] * multiplier[1]
    A1_resized[1, 1] = A1[1, 1] * multiplier[0]
    A1_resized[0, 2] = A1[0, 2] * multiplier[1]
    A1_resized[1, 2] = A1[1, 2] * multiplier[0]

    A2_resized = np.copy(A2)
    A2_resized[0, 0] = A2[0, 0] * multiplier[1]
    A2_resized[1, 1] = A2[1, 1] * multiplier[0]
    A2_resized[0, 2] = A2[0, 2] * multiplier[1]
    A2_resized[1, 2] = A2[1, 2] * multiplier[0]

    mult1 = -np.matmul(ext1[:,:3], ext1[:,3])
    mult2 = -np.matmul(ext2[:, :3], ext2[:, 3])
    ext1_1 = np.concatenate((ext1[:,:3], np.reshape(mult1, (3,1))), axis = 1)
    ext2_1 = np.concatenate((ext2[:,:3], np.reshape(mult2, (3,1))), axis = 1)
    Po1 = np.matmul(A1_resized, ext1)
    Po2 = np.matmul(A2_resized, ext2)

    # Steps followed in the paper: "A compact algorithm for rectification of stereo pairs"

    # Optical centers
    c1 = -np.matmul(np.linalg.inv(Po1[:, 0:3]), Po1[:, 3])
    c2 = -np.matmul(np.linalg.inv(Po2[:, 0:3]), Po2[:, 3])

    # New X axis
    v1 = c2 - c1
    # New Y axis
    v2 = np.cross(R1[2, :].T, v1)
    # New Z axis
    v3 = np.cross(v1, v2)

    # New extrinsic parameters
    R = np.array([v1.T / np.linalg.norm(v1), v2.T / np.linalg.norm(v2), v3.T / np.linalg.norm(v3)])
    # translation is left unchanged

    # New intrinsic parameters
    A = (A1_resized + A2_resized) / 2
    A[0, 1] = 0  # no skew

    # New projection matrices
    Pn1 = np.matmul(A, np.concatenate((R, np.array([np.matmul(-R, c1)]).T), axis=1))
    Pn2 = np.matmul(A, np.concatenate((R, np.array([np.matmul(-R, c2)]).T), axis=1))

    # New extrinsic parameters
    ext1_rect = np.matmul(np.linalg.inv(A), Pn1)
    ext2_rect = np.matmul(np.linalg.inv(A), Pn2)

    # Rectifying image transformation
    T1 = np.matmul(Pn1[0:3, 0:3], np.linalg.inv(Po1[0:3, 0:3]))
    T2 = np.matmul(Pn2[0:3, 0:3], np.linalg.inv(Po2[0:3, 0:3]))

    T1 = T1 / T1[2, 2]
    T2 = T2 / T2[2, 2]

    # Unrectifying image transformation
    T1_inv = np.matmul(Po1[0:3, 0:3], np.linalg.inv(Pn1[0:3, 0:3]))
    T2_inv = np.matmul(Po2[0:3, 0:3], np.linalg.inv(Pn2[0:3, 0:3]))

    # This operation is done because TF requires a transformation matrix with its element on the 3rd row and 3rd column being 1
    T1_inv = T1_inv / T1_inv[2, 2]
    T2_inv = T2_inv / T2_inv[2, 2]

    # Rectifying vectors for TF function
    T1_vector = [T1_inv[0, 0], T1_inv[0, 1], T1_inv[0, 2], T1_inv[1, 0], T1_inv[1, 1], T1_inv[1, 2], T1_inv[2, 0], T1_inv[2, 1]]
    T2_vector = [T2_inv[0, 0], T2_inv[0, 1], T2_inv[0, 2], T2_inv[1, 0], T2_inv[1, 1], T2_inv[1, 2], T2_inv[2, 0], T2_inv[2, 1]]

    if TF:

        img_rect1 = tf.contrib.image.transform(image1, T1_vector, interpolation='BILINEAR', output_shape = transformed_image_size)
        img_rect2 = tf.contrib.image.transform(image2, T2_vector, interpolation='BILINEAR', output_shape = transformed_image_size)

    else:

        img_rect1 = cv2.warpPerspective(image1, T1, (image1.shape[1],image1.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=(101, 101, 101))
        img_rect2 = cv2.warpPerspective(image2, T2, (image1.shape[1],image1.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=(101, 101, 101))

    # Save rectification parameters if a directory is given
    if directory != None:

        if not os.path.exists(directory):
            os.mkdir(directory)

        print('Saving rectified parameters to ' + directory)
        np.save(os.path.join(directory, 'A.npy'), A)
        np.save(os.path.join(directory, 'A1.npy'), A1_resized)
        np.save(os.path.join(directory, 'A2.npy'), A2_resized)
        np.save(os.path.join(directory, 'ext1.npy'), ext1)
        np.save(os.path.join(directory, 'ext2.npy'), ext2)
        np.save(os.path.join(directory, 'ext1_rect.npy'), ext1_rect)
        np.save(os.path.join(directory, 'ext2_rect.npy'), ext2_rect)
        np.save(os.path.join(directory, 'T1.npy'), T1)
        np.save(os.path.join(directory, 'T2.npy'), T2)
        print(A)

    return img_rect1, img_rect2

def unrectify(image1, image2, A1, A2, ext1, ext2, TF = True, transformed_image_size = (1080,1920), calibration_image_size = (1080,1920)):
    R1 = ext1[0:3, 0:3]

    # # We need to adapt the intrinsics matrix to the change in size we want to get
    multiplier = np.divide(transformed_image_size, calibration_image_size)

    A1_resized = np.copy(A1)
    A1_resized[0, 0] = A1[0, 0] * multiplier[1]
    A1_resized[1, 1] = A1[1, 1] * multiplier[0]
    A1_resized[0, 2] = A1[0, 2] * multiplier[1]
    A1_resized[1, 2] = A1[1, 2] * multiplier[0]

    A2_resized = np.copy(A2)
    A2_resized[0, 0] = A2[0, 0] * multiplier[1]
    A2_resized[1, 1] = A2[1, 1] * multiplier[0]
    A2_resized[0, 2] = A2[0, 2] * multiplier[1]
    A2_resized[1, 2] = A2[1, 2] * multiplier[0]

    Po1 = np.matmul(A1_resized, ext1)
    Po2 = np.matmul(A2_resized, ext2)

    # Steps followed in the paper: "A compact algorithm for rectification of stereo pairs"

    # Optical centers
    c1 = -np.matmul(np.linalg.inv(Po1[:, 0:3]), Po1[:, 3])
    c2 = -np.matmul(np.linalg.inv(Po2[:, 0:3]), Po2[:, 3])

    # New X axis
    v1 = c2 - c1
    # New Y axis
    v2 = np.cross(R1[2, :].T, v1)
    # New Z axis
    v3 = np.cross(v1, v2)

    # New extrinsic parameters
    R = np.array([v1.T / np.linalg.norm(v1), v2.T / np.linalg.norm(v2), v3.T / np.linalg.norm(v3)])
    # translation is left unchanged

    # New intrinsic parameters
    A = (A1_resized + A2_resized) / 2
    A[0, 1] = 0  # no skew

    # New projection matrices
    Pn1 = np.matmul(A, np.concatenate((R, np.array([np.matmul(-R, c1)]).T), axis=1))
    Pn2 = np.matmul(A, np.concatenate((R, np.array([np.matmul(-R, c2)]).T), axis=1))

    # Rectifying image transformation
    T1 = np.matmul(Pn1[0:3, 0:3], np.linalg.inv(Po1[0:3, 0:3]))
    T2 = np.matmul(Pn2[0:3, 0:3], np.linalg.inv(Po2[0:3, 0:3]))

    T1 = T1 / T1[2, 2]
    T2 = T2 / T2[2, 2]

    # Unrectifying image transformation
    T1_inv = np.matmul(Po1[0:3, 0:3], np.linalg.inv(Pn1[0:3, 0:3]))
    T2_inv = np.matmul(Po2[0:3, 0:3], np.linalg.inv(Pn2[0:3, 0:3]))

    # This operation is done because TF requires a transformation matrix with its element on the 3rd row and 3rd column being 1
    T1_inv = T1_inv / T1_inv[2, 2]
    T2_inv = T2_inv / T2_inv[2, 2]

    # Rectifying vectors for TF function
    T1_vector = [T1[0, 0], T1[0, 1], T1[0, 2], T1[1, 0], T1[1, 1], T1[1, 2], T1[2, 0], T1[2, 1]]
    T2_vector = [T2[0, 0], T2[0, 1], T2[0, 2], T2[1, 0], T2[1, 1], T2[1, 2], T2[2, 0], T2[2, 1]]

    if TF:

        img_rect1 = tf.contrib.image.transform(image1, T1_vector, interpolation='BILINEAR', output_shape = transformed_image_size)
        img_rect2 = tf.contrib.image.transform(image2, T2_vector, interpolation='BILINEAR', output_shape = transformed_image_size)
    else:

        img_rect1 = cv2.warpPerspective(image1, T1_inv, (image1.shape[1], image1.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=(101, 101, 101))
        img_rect2 = cv2.warpPerspective(image2, T2_inv, (image1.shape[1], image1.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=(101, 101, 101))

    return img_rect1, img_rect2

