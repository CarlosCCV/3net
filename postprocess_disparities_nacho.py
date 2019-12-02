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
def readParameters(directory, stereoPair):

    # Intrinsic matrices
    A = np.zeros((3,3,2))
    if stereoPair % 2 == 0:
        A[: ,: ,0] = np.load(os.path.join(directory,'A2.npy'))
    else:
        A[:, :, 0] = np.load(os.path.join(directory, 'A1.npy'))

    A[:, :, 1] = np.load(os.path.join(directory,'A.npy'))

    # Extrinsic matrices
    ext = np.zeros((3,4,2))
    ext_other = np.zeros((3,4,2))
    if stereoPair % 2 == 0:
        ext[:,:,0] = np.load(os.path.join(directory,'ext2.npy'))
        ext[:,:,1] = np.load(os.path.join(directory, 'ext2_rect.npy'))
        ext_other[:, :, 0] = np.load(os.path.join(directory, 'ext1.npy'))
        ext_other[:, :, 1] = np.load(os.path.join(directory, 'ext1_rect.npy'))
    else:
        ext[:, :, 0] = np.load(os.path.join(directory, 'ext1.npy'))
        ext[:, :, 1] = np.load(os.path.join(directory, 'ext1_rect.npy'))
        ext_other[:, :, 0] = np.load(os.path.join(directory, 'ext2.npy'))
        ext_other[:, :, 1] = np.load(os.path.join(directory, 'ext2_rect.npy'))

    baseline = np.linalg.norm(ext[:,3,:] - ext_other[:,3,:], axis = 0)

    # Rectifying Transformation matrices
    if stereoPair % 2 == 0:
        T = np.load(os.path.join(directory,'T2.npy'))
    else:
        T = np.load(os.path.join(directory,'T1.npy'))

    return A, ext, baseline, T

# Construct Q matrix, which is used to reproject points to 3D in the OpenCV function reprojectImageTo3D
# Need 4x4 matrix of the form Q = [1, 0, 0, -cx]
#                                 [0, 1, 0, -cy]
#                                 [0, 0, 0,   fx]
#                                 [0, 0, -1/Tx  (cx - cx')/Tx];  being (cx, cy): principal point of the camera aligned, fx: focal distance, Tx: stereo baseline
def constructQmatrix(A, baseline):

    # c (principal point) from intrinsic matrix. c = (cx, cy)
    c = np.squeeze(np.array([A[0:2, 2]]))

    # fx (focal)
    fx = A[0, 0]

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

def projectPoints(points3D, projectionMatrix):
    points_projected = np.zeros((256, 512, 2))
    for i in range(256):
        for j in range(512):
            point3D_hom = [points3D[i,j,0], points3D[i,j,1], points3D[i,j,2], 1]
            points2d_array_hom = np.matmul(projectionMatrix, point3D_hom)
            points_projected[i,j,:] = points2d_array_hom[:2] / points2d_array_hom[2]

    return points_projected

def buildDepthMap(points2D, depth, znear, zfar):
    width = 512
    height = 256
    depth_map = np.zeros((256, 512))
    for i in range(256):
        for j in range(512):
            x = np.round(points2D[i, j, 0]).astype(np.int)
            y = np.round(points2D[i, j, 1]).astype(np.int)
            if (x > 0 and x < width and y>0 and y < height):
                depth_map[y,x] = depth[i,j]

    depth_map = applyZRange(depth_map, znear, zfar)
    return depth_map


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

def depthToDisparity(depthMap, fx, baseline):
    print(depthMap.shape)
    disparity = np.zeros(depthMap.shape)
    width = depthMap.shape[1]

    for i in range(depthMap.shape[0]):
        for j in range(depthMap.shape[1]):
            disparity[i,j] = (baseline * fx) / (depthMap[i,j]*1000 * width)
    disp = (baseline * fx) / (depthMap * width)
    return disparity#,disp


def main(directory, stereo_pairs, znear, zfar, fuse_criteria):
    depth_map_list = []
    A = np.zeros((3,3,2,stereo_pairs))
    ext = np.zeros((3,4,2,stereo_pairs))
    baseline = np.zeros((stereo_pairs,2))
    T = np.zeros((3,3,stereo_pairs))
    for k in range(stereo_pairs):
        # Read calibration and rectification matrices
        A[:,:,:,k], ext[:, :, :, k], baseline[k,:], T[:,:,k] = readParameters(os.path.join(directory, 'stereoPair' + str(k), 'Parameters'), k)

        # Read Disparities
        disparities = np.load(os.path.join(directory, 'stereoPair' + str(k), 'disparity' + str(k) + '.npy'))
        num_images = disparities.shape[0]
        num_images = 2
        height = disparities.shape[1]
        width = disparities.shape[2]

        # Construct Q matrix from rectified intrinsic and extrinsic parameters
        Q = constructQmatrix(A[:,:,1,k], baseline[k,1])

        points3d_source = np.zeros((num_images, height, width, 3))
        points3d_dest = np.zeros((num_images, height, width, 3))
        depth_map_source = np.zeros((num_images, height, width))
        depth_map_dest = np.zeros((num_images, height, width))
        ones_array = np.ones((1, height * width))

        # rotation = ext[:,:3,0,k]
        # translation = np.matmul(-ext[:,:3,0,k], ext[:,3,0,k])
        # translation = np.reshape(translation, (3,1))
        rotation = np.identity(3)
        translation = np.zeros((3,1))
        extrinsics = np.concatenate((rotation, translation), axis = 1)
        projectionMatrix = np.matmul(A[:,:,0,k], extrinsics)
        for i in range(num_images):

            # Obtain 3D points in Camera Coordinates
            points3d_source[i, :, :, :] = cv2.reprojectImageTo3D(np.float32(disparities[i, :, :] * width), np.float32(Q))

            # Change the coordinates system to the desired Camera Coordinates System (central camera)
            points3d_dest[i, :, :, :] = changeCameraCoordinateSystem(points3d_source[i, :, :, :], ext[: ,:, 1, k], ext[:, :, 0, k])

            # Project points
            points_projected = projectPoints(points3d_dest[i, :, :, :], projectionMatrix)

            # Build depth map
            depth_map_dest[i,:,:] = buildDepthMap(points_projected, points3d_dest[i, :, :, 2], znear, zfar)

        depth_map_list.append(depth_map_dest)

    # Convert list to matrix
    depth_maps = np.zeros((num_images, height, width, stereo_pairs))
    for i in range(stereo_pairs):
        depth_maps[:, :, :, i] = depth_map_list[i]

    # Fuse depth maps to obtain an improved one
    fusedDepthMaps = fuseDepthMaps(depth_maps, fuse_criteria)

    plt.figure(0)
    plt.imshow(depth_maps[1,:,:,0], cmap = 'gray')
    plt.figure(1)
    plt.imshow(depth_maps[1, :, :, 1], cmap='gray')
    plt.figure(2)
    plt.imshow(fusedDepthMaps[1, :, :])
    plt.show()

    #### THAT WOULD BE THE RESULT, NOW I NEED TO TRANSFORM IT TO OBTAIN TO DISPARITIES IN THE RECTIFIED CAMERAS, IN ORDER
    #### TO EVALUATE THE QUALITY OF THE DEPTH MAP OBTAINED

    fusedDisparities = np.zeros((fusedDepthMaps.shape[0], fusedDepthMaps.shape[1], fusedDepthMaps.shape[2]))
    fused3DPoints = np.zeros((num_images, height, width, 3))

    fused3DperPair = np.zeros((num_images, height, width, 3, stereo_pairs)) # In appropriate Camera Coordinates

    print(fusedDisparities.shape)

    # # Get disparity
    # # Construct Q matrix from intrinsic and extrinsic parameters of the central camera
    # Q = constructQmatrix(A[:, :, 0, 0], baseline[0, 0])
    #
    # # Change from depth to disparity
    # fusedDisparities[:, :, :] = depthToDisparity(fusedDepthMaps, A[0, 0, 0, 0], baseline[0, 0])
    #
    # # Obtain 3D points
    # for i in range(num_images):
    #     fused3DPoints[i, :, :, :] = cv2.reprojectImageTo3D(np.float32(fusedDisparities[i, :, :] * width),
    #                                                       np.float32(Q))
    #
    # fusedDepthPerPair = np.zeros((num_images, height, width, stereo_pairs))
    # fusedDisparitiesPerPair = np.zeros((num_images, height, width, stereo_pairs))
    # for k in range(stereo_pairs):
    #     for i in range(num_images):
    #
    #         # Change the coordinates system to the desired Camera Coordinates System (central camera)
    #         fused3DperPair[i, :, :, :, k] = changeCameraCoordinateSystem(fused3DPoints[i, :, :, :], ext[:, :, 0, k],
    #                                                                  ext[:, :, 1, k])

    #         # Change perspective of the depth map tobe aligned with the desired camera
    #         fusedDepthPerPair[i, :, :, k] = changePerspective(fused3DperPair[i, :, :, 2, k], np.linalg.inv(T[:, :, k]))
    #
    #         # Obtain depth map and apply zrange
    #         fusedDepthPerPair[i, :, :, k] = applyZRange(fusedDepthPerPair[i, :, :, k], znear, zfar)
    #
    #     # Change from depth to disparities
    #     fusedDisparitiesPerPair[:, :, :, k] = depthToDisparity(fusedDepthPerPair[:, :, :, k], A[0, 0, 1, k], baseline[k, 1])






    # idx = np.isnan(fusedDisparities[:,:,:])
    # fusedDisparities[idx] = 0
    # print(np.amax(fusedDisparities))
    # plt.figure(0)
    # for i in range(256):
    #     for j in range(512):
    #         if (fusedDisparitiesPerPair[1,i,j,0] < znear or fusedDisparitiesPerPair[1,i, j,0] > 1):
    #             fusedDisparitiesPerPair[1,i, j,0] = 0
    # plt.imshow(fusedDisparitiesPerPair[1,:,:,0], cmap='gray')
    # plt.figure(1)
    # plt.imshow(fusedDisparitiesPerPair[1,:,:,1], cmap = 'gray')
    # plt.figure(2)
    #
    # plt.imshow(fusedDisparities[1, :, :], cmap = 'gray')
    # plt.show()

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
