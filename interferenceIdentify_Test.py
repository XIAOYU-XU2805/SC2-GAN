import argparse

from os import listdir
from os.path import join
import os.path
import scipy.io
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from glob import glob

from model_xxy_SC2GAN import AEGenerator1
import numpy as np
from utils1 import double_channel, single_channel, data_abs_2d

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--data_path_r', default='./inputs_real_data/', type=str, help='test low resolution image name')  
parser.add_argument('--data_path_i', default='./inputs_imag_data/', type=str, help='test low resolution image name')
parser.add_argument('--model_name', default='./netG_epoch_1_0_0_39.pth', type=str, help='generator model epoch name')

parser.add_argument('--dropout', default=0.5, type=float)
opt = parser.parse_args()

TEST_MODE = True if opt.test_mode == 'GPU' else False
dataset_dir = opt.data_path_r
MODEL_NAME = opt.model_name

# 读取测试数据

test_noisy_r = glob(
    os.path.join("./inputs_real_data/", '*.mat'))  # input_frame_pattern应该为*.txt
test_noisy_i = glob(
    os.path.join("./inputs_imag_data/", '*.mat'))
test_noisy = list(zip(test_noisy_r, test_noisy_i))
print("test_noisy path list shape:.{}".format(np.shape(test_noisy)))

model = AEGenerator1()

if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load(MODEL_NAME, map_location=lambda storage, loc: storage), strict=False)
else:
    model.load_state_dict(torch.load(MODEL_NAME))

model.eval()

with torch.no_grad():
    out_list = []
    images_list = []
    for index in range(99, 99+len(test_noisy)):  #len(image_filenames)
        print("index:{}".format(index))
        test_noisy_ = double_channel(opt.data_path_r+str(index+1)+'.mat', opt.data_path_i+str(index+1)+'.mat', False)  
        test_noisy_ = np.array(test_noisy_[0]).astype(np.float32)
        test_noisy_ = torch.tensor(test_noisy_)
        test_noisy_ = torch.unsqueeze(test_noisy_, 3)
        test_noisy_ = test_noisy_.permute(3, 2, 0, 1)
        test_noisy_ = test_noisy_.cuda()

        ######## SC2-GAN use
        test_results, test_results_sg = model(test_noisy_)
        test_results = torch.squeeze(test_results.permute(0, 2, 3, 1))
        test_results_sg = test_results_sg.permute(0, 2, 3, 1)
        test_results = torch.Tensor.cpu(test_results).numpy()
        test_results_sg = torch.Tensor.cpu(test_results_sg).numpy()
        scipy.io.savemat('./{}.mat'.format(index + 1), {'output': test_results, 'mask': test_results_sg})

