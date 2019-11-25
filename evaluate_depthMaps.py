import numpy as np
import argparse
import os
import tensorflow as tf
def readParameters(directory):

    # Intrinsic matrices
    A = np.zeros((3,3))
    A[:,:] = np.load(os.path.join(directory,'A.npy'))
    fx = A[0,0]

    # Extrinsic matrices
    ext = np.zeros((3,4,2))
    ext[:,:,0] = np.load(os.path.join(directory,'ext1.npy'))
    ext[:,:,1] = np.load(os.path.join(directory, 'ext2.npy'))
    baseline = np.linalg.norm(ext[:,3,0] - ext[:,3,1])
    return fx, baseline

def depthToDisparity(depthMap, fx, baseline):
    print(depthMap.shape)
    disparity = np.zeros(depthMap.shape)
    width = depthMap.shape[1]

    for i in range(depthMap.shape[0]):
        for j in range(depthMap.shape[1]):
            disparity[i,j] = (baseline * fx) / (depthMap[i,j] * width)

    return disparity

def main(directory):

    # Load the postprocessed depth maps
    depthMaps = np.load(os.path.join(directory, 'fusedDepthMaps.npy'))
    print(depthMaps.shape)

    # Load parameters associated with the stereo pair demanded
    fx, baseline = readParameters(directory)

    # Convert from depth maps to disparities
    disparity
    # Use bilinear sampler to generate the other image

    # Compare with the original image

if __name__=="__main__":

    # Directory example:
    # directory = "/mnt/goatse/DATASETS/FVV/Models/fvv_trinet_LRConsistency_8_masks"

    # Parser
    parser = argparse.ArgumentParser(description="Trinet post-processing.")
    parser.add_argument('--directory', type=str, help='Directory containing trinet testing data and .npy rectification and calibration parameters.', required=True)
    parser.add_argument('--stereo_pair', type=int, help='Number of the stereo pair to be evaluated.', required= True)
    args = parser.parse_args()

    main(args.directory)
