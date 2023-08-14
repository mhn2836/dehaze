import os
import torch

import loss
import network
import dataset
import argparse
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
from loss import *
from torchvision.models import vgg16
from kornia.filters import laplacian

#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def train(args):
    model = network.B_transformer()
    model.load_state_dict(torch.load("model/dehaze.pth"))
    model = model.cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    
    
    mae = nn.L1Loss().cuda()
    new_loss = loss.MS_SSIM_L1_LOSS().cuda()
    smoothl1 = nn.SmoothL1Loss().cuda()
    ssim_loss = loss.SSIM_loss().cuda()

    vgg_model = vgg16(pretrained=True).features[:16]
    vgg_model = vgg_model.to('cuda')
    for param in vgg_model.parameters():
        param.requires_grad = False

    # per loss
    loss_network = LossNetwork(vgg_model).cuda()
    loss_network.eval()


    content_folder1 = 'haze'
    information_folder = 'gt'

    train_loader = dataset.style_loader(content_folder1, information_folder, args.size, args.batch_size)
    
    num_batch = len(train_loader)

    for epoch in range(args.epoch):
      for idx, batch in tqdm(enumerate(train_loader), total=num_batch):
            total_iter = epoch*num_batch  + idx
               
            content = batch[0].float().cuda()
            information = batch[1].float().cuda()
            
            optimizer.zero_grad()

            output = model(content)         
                     
            # total_loss =  mse(output , information)
            # total_loss = new_loss(output, information)

            # per_loss + l1
            #smooth_loss = F.smooth_l1_loss(output , information)
            #perceptual_loss = loss_network(output , information)
            #total_loss = 1 * smooth_loss + 0.04 * perceptual_loss

            # perceptual_loss = ploss(output , information)
            # total_loss = smooth_loss


            # multi loss
            smooth = mae(output , information)
            # ssim = ssim_loss(output , information)
            # perceptual_loss = loss_network(output, information)

            # total_loss = 2 * smooth + 1 * ssim + 1 * perceptual_loss

            total_loss = smooth

            total_loss.backward()      

            optimizer.step()
            
            if np.mod(total_iter+1, 1) == 0:
                print('{}, Epoch:{} Iter:{} total loss: {}'.format(args.save_dir, epoch, total_iter, total_loss.item()))
                          
            if not os.path.exists(args.save_dir+'/image'):
                os.mkdir(args.save_dir+'/image')

      if epoch >= 80:
        if epoch % 20 == 0:
            out_image = torch.cat([content[0:3], output[0:3], information[0:3]], dim=0)
            save_image(out_image, args.save_dir+'/image/iter{}_1.jpg'.format(total_iter+1))
            torch.save(model.state_dict(), 'model' +'/our_deblur{}.pth'.format(epoch))
        elif epoch >= 390 and epoch % 3 == 0:
            out_image = torch.cat([content[0:3], output[0:3], information[0:3]], dim=0)
            save_image(out_image, args.save_dir + '/image/iter{}_1.jpg'.format(total_iter + 1))
            torch.save(model.state_dict(), 'model' + '/our_deblur{}.pth'.format(epoch))
    ''''''




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--epoch', default=400, type=int)
    parser.add_argument('--size', default=512, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--save_dir', default='result', type=str)
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    
    train(args)
