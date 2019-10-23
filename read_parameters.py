import os
import numpy as np

def read_calibration_parameters(dataset_path, filenames_file):
    dataset_path = '/mnt/goatse/DATASETS/FVV'
    file_path = os.path.join(dataset_path, 'Calibration', 'Results','extrinsicsAndIntrinsicsComputed.txt')
    file = open(file_path, 'r')
    filenames = open(filenames_file, 'r')
    lines = filenames.readlines()
    filenames_list = lines[0].split()
    cam_left = filenames_list[0].split('/')[-2]
    cam_center = filenames_list[1].split('/')[-2]
    cam_right = filenames_list[2].split('/')[-2]

    cameras_used = [int(cam_left), int(cam_center), int(cam_right)]
    contents = file.readlines()


    cameras_index_intext = [int(contents.index(s)) for s in contents if "Camera" in s]

    # Intrinsics and Extrinsics arrays (3rd dimension indicates the number of the camera)
    intrinsics = np.zeros((3,3,3))
    extrinsics = np.zeros((3,4,3))
    for i,cam in enumerate(cameras_used):

        intrinsics[0, :, i] = [float(j) for j in contents[cameras_index_intext[cam] + 2].split()]
        intrinsics[1, :, i] = [float(j) for j in contents[cameras_index_intext[cam] + 3].split()]
        intrinsics[2, :, i] = [float(j) for j in contents[cameras_index_intext[cam] + 4].split()]

        extrinsics[0, :, i] = [float(j) for j in contents[cameras_index_intext[cam] + 7].split()]
        extrinsics[1, :, i] = [float(j) for j in contents[cameras_index_intext[cam] + 8].split()]
        extrinsics[2, :, i] = [float(j) for j in contents[cameras_index_intext[cam] + 9].split()]

    return intrinsics, extrinsics


def main():
    dataset_path = '/mnt/goatse/DATASETS/FVV'
    intrinsics, extrinsics = read_calibration_parameters(dataset_path)
    for i in range(intrinsics.shape[2]):
        print("Camera " + str(i))
        print(intrinsics[:, :, i])
        print(extrinsics[:, :, i])
        print("\n")


if __name__=="__main__":
    main()