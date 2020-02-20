#!/usr/bin/env python3

from PIL import Image
from glob import glob
import numpy as np
import os
import scipy.misc

folder = 'valid/kodak/'
#folder = 'valid/raise/'
img_format = '.png'
#folder = 'valid/clicval_m_768_jpg/'
#img_format = '.jpg'
start = len(folder)
end = -len(img_format)

for image_file in glob(folder + '*' + img_format):
    print(image_file)

    os.system('python bls2017_joint_gmm.py --num_filters 128 --checkpoint_dir train_joint_0.015_MSE_128filter_3x3(4)_v1.0_subpixel_OpenImage_GMM --verbose compress ' \
              + image_file + ' temp/' + image_file[start:end] + '_recon.npz')
    #os.system('python bls2017_hyper_3x3_subpixel_128x128_lossless_v21.py --num_filters 128 decompress temp/' \
    #          + image_file[12:-4] + '_recon.npz temp/' + image_file[12:-4] + '_recon.png')
