# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 00:16:07 2021
@author: xiuzhang

参考资料：
https://blog.csdn.net/leviopku/article/details/83898619
"""

import sys
import cv2
import math
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import time

import torch

'''
def DarkChannel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark


def AtmLight(im, dark):
    [h, w] = im.shape[:2]
    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    darkvec = dark.reshape(imsz, 1)
    imvec = im.reshape(imsz, 3)

    indices = darkvec.argsort()
    indices = indices[imsz - numpx::]

    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A


def TransmissionEstimate(im, A, sz):
    omega = 0.95
    im3 = np.empty(im.shape, im.dtype)

    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]

    transmission = 1 - omega * DarkChannel(im3, sz)
    return transmission


def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * im + mean_b
    return q


def TransmissionRefine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    r = 60
    eps = 0.0001
    t = Guidedfilter(gray, et, r, eps)

    return t


def Recover(im, t, A, tx=0.1):
    res = np.empty(im.shape, im.dtype)
    t = cv2.max(t, tx)

    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]

    return res
'''

def guided_filter(I,p,win_size,eps):

    mean_I = cv2.blur(I,(win_size,win_size))
    mean_p = cv2.blur(p,(win_size,win_size))

    corr_I = cv2.blur(I*I,(win_size,win_size))
    corr_Ip = cv2.blur(I*p,(win_size,win_size))

    var_I = corr_I-mean_I*mean_I
    cov_Ip = corr_Ip - mean_I*mean_p

    a = cov_Ip/(var_I+eps)
    b = mean_p-a*mean_I

    mean_a = cv2.blur(a,(win_size,win_size))
    mean_b = cv2.blur(b,(win_size,win_size))

    q = mean_a*I + mean_b
    return q
def get_min_channel(img):
    return np.min(img,axis=2)
def min_filter(img,r):
    kernel = np.ones((2*r-1,2*r-1))
    return cv2.erode(img,kernel)#最小值滤波器，可用腐蚀替代
def get_A(img_haze,dark_channel,bins_l):
    hist,bins = np.histogram(dark_channel,bins=bins_l)#得到直方图
    d = np.cumsum(hist)/float(dark_channel.size)#累加
    # print(bins)
    threshold=0
    for i in range(bins_l-1,0,-1):
        if d[i]<=0.999:
            threshold=i
            break
    A = img_haze[dark_channel>=bins[threshold]].max()
    #候选区域可视化
    show  = np.copy(img_haze)
    show[dark_channel>=bins[threshold]] = 0,0,255
    cv2.imwrite('./most_haze_opaque_region.jpg',show*255)
    return A
def get_t(img_haze,A,t0=0.1,w=0.95):
    out = get_min_channel(img_haze)
    out = min_filter(out,r=7)
    t = 1-w*out/A #需要乘上一系数w，为远处的物体保留少量的雾
    t = np.clip(t,t0,1)#论文4.4所提到t(x)趋于0容易产生噪声，所以设置一最小值0.1
    return t



if __name__ == '__main__':
    path = r'./sots/'

    outdir = 'C:/Users\MHN\Desktop\DCP_train//'

    for image in os.listdir(path):
        print(image[:-4])

        src = cv2.imread(path + image)

        src = cv2.resize(src, (512, 512))

        I = src.astype('float64') / 255

        start = time.time()

        dark_channel = get_min_channel(I)
        dark_channel_1 = min_filter(dark_channel, r=7)
        # cv2.imwrite("./dark_channel.jpg", dark_channel_1*255)

        A = get_A(I, dark_channel_1, bins_l=2000)

        t = get_t(I, A)
        t = guided_filter(dark_channel, t, 81, 0.001)
        t = t[:, :, np.newaxis].repeat(3, axis=2)  # 升维至(r,w,3)

        J = (I - A) / t + A

        J = np.clip(J, 0, 1)
        J = J * 255
        J = np.uint8(J)

        end = time.time()
        print("FP32 Iterations per second: ", 1 / (end - start))

        # cv2.imwrite(outdir + image[:-4] + '.png', J)
