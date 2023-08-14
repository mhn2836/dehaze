# 
This is the PyTorch implementation for my work based on CVPR'21 paper 4KDehazing. 
The model can removal hazy, smoke.




Setup:

依赖的库
torch, numpy, tqdm, torchvision, kornia, opencv-python


Training：

将带雾训练数据集放在./haze 文件夹下 对应的清晰数据集放在./gt文件夹下。
运行命令 python train.py。 
训练过程可在./result文件夹下找到。
模型保存在./model文件夹下。

Test model：

提供自己基于部分OTS数据集训练的模型dehaze.pth
将需要测试的数据集放在./test文件下。
运行命令 python test_model.py。
自行建立test_res文件夹，测试结果可在./test_res文件夹下找到。

SSIM_PSNR Test：

ssim_psnr.py

Time inference：

time.py

Cite:
{
  title     = {Ultra-High-Definition Image Dehazing via Multi-Guided Bilateral Learning},
  booktitle = {CVPR},
  year      = {2021}
}






