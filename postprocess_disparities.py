import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
import math
import argparse

# Read the intrinsics and extrinsics parameters of both original and rectified cameras
# kept in memory after training the Trinet Model
# Returns:
# A matrix: matrix 3x3x3 containing intrinsic matrices for:
#       1st dimension: Original left intrinsic matrix
#       2nd dimension: Original right intrinsic matrix
#       3rd dimension: Rectified intrinsic matrix (right now same matrix for both cameras)
# Ext matrix: matrix 3x4x4 containing extrinsic matrices for:
#       1st dimension: Original left extrinsic matrix
#       2nd dimension: Original right extrinsic matrix
#       3rd dimension: Rectified left extrinsic matrix
#       4th dimension: Rectified right extrinsic matrix
# Rectifying transformation matrix: 3x3x2 rectifying transformation matrices used to transform images fed to Trinet
#   rectified_point = T * originalPoint
#       1st dimension: T matrix from original left to rectified
def readParameters(directory):

    # Intrinsic matrices
    A = np.zeros((3,3,3))
    A[:,:,0] = np.load(os.path.join(directory,'A1.npy'))
    A[:,:,1] = np.load(os.path.join(directory,'A2.npy'))
    A[:,:,2] = np.load(os.path.join(directory,'A.npy'))

    # Extrinsic matrices
    ext = np.zeros((3,4,4))
    ext[:,:,0] = np.load(os.path.join(directory,'ext1.npy'))
    ext[:,:,1] = np.load(os.path.join(directory, 'ext2.npy'))
    ext[:,:,2] = np.load(os.path.join(directory, 'ext1_rect.npy'))
    ext[:,:,3] = np.load(os.path.join(directory, 'ext2_rect.npy'))


    # Rectifying Transformation matrices
    T = np.zeros((3,3,2))
    T[:,:,0] = np.load(os.path.join(directory,'T1.npy'))
    T[:,:,1] = np.load(os.path.join(directory,'T2.npy'))

    return A, ext, T

# Construct Q matrix, which is used to reproject points to 3D in the OpenCV function reprojectImageTo3D
# Need 4x4 matrix of the form Q = [1, 0, 0, -cx]
#                                 [0, 1, 0, -cy]
#                                 [0, 0, 0,   fx]
#                                 [0, 0, -1/Tx  (cx - cx')/Tx];  being (cx, cy): principal point of the camera aligned, fx: focal distance, Tx: stereo baseline
def constructQmatrix(A, ext):

    # c (principal point) from intrinsic matrix. c = (cx, cy)
    c = np.squeeze(np.array([A[0:2, 2, 2]]))

    # fx (focal)
    fx = A[0, 0, 0]

    # Translation vectors
    t1 = ext[:, 3, 0]
    t2 = ext[:, 3, 1]

    # Baseline
    baseline = np.linalg.norm(t1 - t2)

    # Q matrix
    Q = np.array([[1, 0, 0, -c[0]], [0, 1, 0, -c[1]], [0, 0, 0, fx], [0, 0, -1 / baseline, 0]])

    return Q

# This function changes form one camera coordinate system to another one by using Hartley formulation:
#       1st: Converts form 1st Camera Coordinate system to World Coordinate System
#       2nd: Converts from World Coordinate System to 2nd Camera Coordinate System
def changeCameraCoordinateSystem(point3D_source, ext_source, ext_dest):
    point3D_dest = np.zeros(point3D_source.shape)

    for i in range(256):
        for j in range(512):
            # Convert to world coordinates
            point_cl = np.array([point3D_source[i,j,0], point3D_source[i,j,1], point3D_source[i,j,2]])
            point_world = np.matmul(np.linalg.inv(ext_source[:,:3]), (point_cl - ext_source[:,3]))
            point_world_hom = np.array([point_world[0], point_world[1], point_world[2], 1])

            # Convert to destination camera coordinates
            point_centralCoord = np.matmul(ext_dest, point_world_hom)

            point3D_dest[i,j,0] = point_centralCoord[0]
            point3D_dest[i, j, 1] = point_centralCoord[1]
            point3D_dest[i, j, 2] = point_centralCoord[2]

    return point3D_dest


def changePerspective(image, T):
    T_inv = np.linalg.inv(T)
    image_dest = cv2.warpPerspective(image, T_inv, (512, 256), borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(101, 101, 101))
    return image_dest


def depthToMeters(depth_map):
    depth_map = depth_map / 1000

    if np.nansum(depth_map) < 0:
        depth_map = depth_map * -1

    return depth_map


def applyZRange(image, znear, zfar):

    image = depthToMeters(image)
    for i in range(256):
        for j in range(512):
            if (image[i,j] < znear or image[i, j] > zfar):
                image[i, j] = 0

    return image

def fuseDepthMaps(depth_maps, criteria = 'mean'):

    if criteria == 'mean':
        # Mean over the elements different from zero
        difZerosMatrix = (depth_maps != 0).astype(int)
        difZerosSum = np.sum(difZerosMatrix, axis = 3)

        depthMapsSum = np.sum(depth_maps, axis = 3)
        fused_depth = np.divide(depthMapsSum, difZerosSum)

    elif criteria == 'near':
        depth_maps[np.where(depth_maps == 0)] = math.nan
        fused_depth = np.nanmin(depth_maps, axis = 3)

    else:
        sys.exit("No criteria for fusing depth maps given. Need to specify 'mean' or 'near' criteria.")

    # Remove nan
    fused_depth[np.where(fused_depth == math.nan)] = 0
    return fused_depth

def main(directory, stereo_pairs, znear, zfar, fuse_criteria):
    depth_map_list = []
    for k in range(stereo_pairs):
        # Read calibration and rectification matrices
        A, ext, T = readParameters(os.path.join(directory, 'stereoPair' + str(k), 'Parameters'))

        # Read Disparities
        disparities = np.load(os.path.join(directory, 'stereoPair' + str(k), 'disparity' + str(k) + '.npy'))
        num_images = disparities.shape[0]
        num_images = 3
        height = disparities.shape[1]
        width = disparities.shape[2]

        # Construct Q matrix from rectified intrinsic and extrinsic parameters
        Q = constructQmatrix(A, ext[:,:,2:])

        points3d_source = np.zeros((num_images, height, width, 3))
        points3d_dest = np.zeros((num_images, height, width, 3))
        depth_map_source = np.zeros((num_images, height, width))
        depth_map_dest = np.zeros((num_images, height, width))
        for i in range(num_images):

            # Obtain 3D points in Camera Coordinates
            points3d_source[i, :, :, :] = cv2.reprojectImageTo3D(np.float32(disparities[i, :, :] * width), np.float32(Q))

            # Change the coordinates system to the desired Camera Coordinates System (central camera)
            points3d_dest[i, :, :, :] = changeCameraCoordinateSystem(points3d_source[i, :, :, :], ext[:,:,3], ext[:,:,1])

            # Obtain depth map and apply zrange
            depth_map_source[i, :, :] = applyZRange(points3d_dest[i, :, :, 2], znear, zfar)

            # Change perspective of the depth map tobe aligned with the desired camera
            depth_map_dest[i, :, :] = changePerspective(points3d_dest[i, :, :, 2], T[:, :, 1])

            # Obtain depth map and apply zrange
            depth_map_dest[i, :, :] = applyZRange(depth_map_dest[i, :, :], znear, zfar)

        depth_map_list.append(depth_map_dest)

    # Convert list to matrix
    depth_maps = np.zeros((num_images, height, width, stereo_pairs))
    for i in range(stereo_pairs):
        depth_maps[:, :, :, i] = depth_map_list[i]

    # Fuse depth maps to obtain an improved one
    fusedDepthMaps = fuseDepthMaps(depth_maps, fuse_criteria)

    # Deberia volver a alinear este mapa de profundidad mejorado con cada uno de los pares estereo

    plt.figure(0)
    plt.imshow(depth_map_list[0][1, :, :], cmap = 'gray')
    plt.figure(1)
    plt.imshow(depth_map_list[1][1, :, :], cmap = 'gray')
    plt.figure(2)
    plt.imshow(fusedDepthMaps[1, :, :], cmap = 'gray')
    plt.show()

if __name__=="__main__":

    # Directory example:
    # directory = "/mnt/goatse/DATASETS/FVV/Models/fvv_trinet_LRConsistency_8_masks"

    # Parser
    parser = argparse.ArgumentParser(description="Trinet post-processing.")
    parser.add_argument('--directory', type=str, help='Directory containing trinet testing data and .npy rectification and calibration parameters.', required=True)
    parser.add_argument('--num_cams', type=int, help='Number of cameras used for training, including the central camera considered. Trinet model uses 3 cameras.', required=True)
    parser.add_argument('--znear', type=int, help='The nearest appreciable depth.', required=True)
    parser.add_argument('--zfar', type=int, help='The farthest appreciable depth.', required=True)
    parser.add_argument('--fuse_criteria', type=str, help="Criteria to fuse the resulting depth maps. 'mean' or 'near' are the possibilities")
    # Future argument (when multiview rectification possible)
    #parser.add_argument('--multiview_rectification', help='Indicates that all the cameras where rectified to the same plane before feeding the network', action='store_true')
    args = parser.parse_args()

    main(args.directory, args.num_cams - 1, args.znear, args.zfar, args.fuse_criteria)
