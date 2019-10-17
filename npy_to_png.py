import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

testing_dir = "/media/ccv/goatse/DATASETS/FVV/Models/fvv_trinet_model/"

disp = np.load(os.path.join(testing_dir,'disparities.npy'))

os.mkdir(os.path.join(testing_dir,'PNG'))


for i in range(1):

    disparity = disp[0,...]
    print(disparity.shape)
    cv2.imwrite('disp_{}.png'.format(i), disparity)