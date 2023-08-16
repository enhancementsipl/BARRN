import cv2
import imageio
from utils import util
import os
import numpy as np

img_list = os.listdir('/home/gyq/data_set/LIVE1/HQ')
Psnr = []
Ssim = []
for name in img_list:
    na=name.split('.')
    SR1 = imageio.imread('/home/gyq/data_set/LIVE1/HQ/' + name)
    SR2 = imageio.imread('/home/gyq/share/gyq/program/02-ar/SRFBN_mix_test/results/SR/SRFBN/LIVE1/x10/'
                         + name.split('.')[0]+'.bmp')
    psnr, ssim = util.calc_metrics(SR1, SR2)
    Psnr.append(psnr)
    Ssim.append(ssim)
print(np.mean(np.array(Psnr)), np.mean(np.array(Ssim)))

# SR1=cv2.imread('/home/gyq/program/SRFBN_CVPR19/results/SR/Set5/SRFBN/x10/step1/baby.bmp')
# SR2=cv2.imread('/home/gyq/program/SRFBN_CVPR19/results/SR/Set5/SRFBN/x10/step5/baby.bmp')
# psnr, ssim = util.calc_metrics(SR1, SR2)
# print(psnr ,ssim)
