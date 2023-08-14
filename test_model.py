import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import network
import numpy as np
import cv2

from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import torch 
import numpy as np
#from skimage.measure import compare_ssim
#from skimage.measure import compare_psnr
from tqdm import tqdm
import kornia
import dataset
from torch.nn import functional as F
from torchvision.utils import save_image
import network
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

my_model = network.B_transformer().to(device)
my_model.eval()
my_model.to(device)

my_model.load_state_dict(torch.load("model/dehaze.pth"))
# to_pil_image = transforms.ToPILImage()


tfs_full = transforms.Compose([
            #transforms.Resize(1080),
            transforms.ToTensor()
        ])

def load_simple_list(src_path):
    name_list = list()
    for name in os.listdir(src_path):
        path = os.path.join(src_path, name)
        name_list.append(path)
    name_list = [name for name in name_list if '.png' or '.jpg' in name]
    name_list.sort()
    return name_list

content_folder1 = 'test'
list_s = load_simple_list(content_folder1)
#list_s = load_simple_list('hazy_sots')

index = 0

for img_path in list_s:
     #image_in = Image.open('/home/dell/4Kdehaze/OHAZE_test/27_outdoor_hazy.jpg').convert('RGB')

     # image_in = Image.open('OHAZE_test/BJ_Google_401.png').convert('RGB')
     #print(img_path)

     image_in = Image.open(img_path).convert('RGB')

     full = tfs_full(image_in).unsqueeze(0).to(device)


     with torch.no_grad():
        output = my_model(full)


     print(index)
     index += 1

     save_image(output[0:3], 'test_res/{}'.format(os.path.basename(img_path))[:-4] + '.png')

