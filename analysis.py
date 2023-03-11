import os
import torch
import random
import os.path as path
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import PySimpleGUI as sg
from einops import rearrange
from config import DAFNet_synapse_version1_cls9_v3 as hyper
from trainer import Trainer_synapse

net = Trainer_synapse("D:\\Projects\\datasets\\Synapse_npy",hyper)

save_path = hyper['save_path']
result_path = os.path.join(save_path, 'analysis')
if os.path.exists(result_path) is False:
  os.mkdir(result_path)

params = torch.load("D:\Projects\\results\DAFNet\\version1_cls_9_v3\\best_epoch.pth")
hyper = params['hyper']
labels = params['labels']
model = net.model.to('cpu')
model.load_state_dict(params['state_dict'])
val_samples = params['validate_samples']
test_samples = net.test_list

# with open("D:\Projects\datasets\Synapse_npy\\test_slice\\test_slice_list.csv") as f:
#    for line in f:
#       x,y = line.split(" ")
#       test_samples.append((x,y[:-1]))

colors = np.asarray([
  [255,0,0],[0,255,0],[0,0,255],
  [255,255,0],[255,0,255],[0,255,255],
  [0,128,255],[128,0,255],[128,255,0]
]).astype(np.float32)

for x_path, y_path in test_samples:
  image = np.load(x_path)
  label = np.load(y_path)
  x = ndimage.zoom(image,224/512.)
  y = ndimage.zoom(label,224/512.,order=0)
  x_ = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
  y_ = model(x_)
  save_name = os.path.basename(x_path).split('.')[0] + ".png"
  bg = np.zeros((224,224,3),np.float32)
  bg_y = np.zeros((224,224,3),np.float32)
  threshold = 0.5
  for i in range(1, 9): # organs 0-9 0: background
     mask = y_[0][i].detach().numpy()
     mask[mask < threshold] = 0
     mask[mask >= threshold] = 1 
     mask_gt = y == i
     fg = np.ones((224, 224, 3)) * colors[i]
     fg_gt = np.ones((224,224,3)) * colors[i]
     for i in range(3):
        fg[:,:,i] = fg[:,:,i] * mask
        fg_gt[:,:,i] = fg_gt[:,:,i] * mask_gt
     bg += fg
     bg_y += fg_gt
     for i in range(3):
        bg[:,:,i] = x * (1. - bg[:,:,i])
        bg_y[:,:,i] = x * (1. - bg_y[:,:,i])
     
  print(np.unique(y))
  xx = cv.cvtColor(x, cv.COLOR_GRAY2RGB)
  img = np.concatenate([xx,bg_y,bg], axis=1)
  cv.imshow("image", img)
  cv.waitKey(-1)
  # fig, ax  = plt.subplots()
  # ax.imshow(img)
  # ax.set_title("image; ground truth; model output")
  # fig.savefig(os.path.join(result_path, save_name))


def analysis_validate():
  for (x_path, y_path) in val_samples:
      ground_img = np.load(x_path)
      x = ndimage.zoom(ground_img, 224/512)
      ground_truth = np.load(y_path)
      y = ndimage.zoom(ground_truth, 224/512, order=0)
      # cv.imshow('win_image',x)
      # cv.imshow('win_label',y)
      x_ = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
      y_ = model(x_) # B C H W

      save_name = os.path.basename(x_path).split('.')[0] + ".png"

      bg = np.zeros((224,224,3),np.float32)
      bg_y = np.zeros((512,512,3),np.float32)
      threshold = 0.5
      for i in range(1, y_.shape[1]): # organs 0-9 0: background
          mask = y_[0][i].detach().numpy()
          mask[mask < threshold] = 0
          mask[mask >= threshold] = 1.

          mask_gt = ground_truth == float(i)

          fg = np.ones((224, 224, 3)) * colors[i]
          fg_gt = np.ones((512,512,3)) * colors[i]
          for i in range(3):
             fg[:,:,i] = fg[:,:,i] * mask
             fg_gt[:,:,i] = fg_gt[:,:,i] * mask_gt

          bg += fg
          bg_y += fg_gt

      for i in range(3):
         bg[:,:,i] = x * (1. - bg[:,:,i])
         bg_y[:,:,i] = ground_img * (1. - bg_y[:,:,i])



      bg_y = cv.resize(bg_y,(224,224),interpolation=cv.BORDER_CONSTANT)
      # cv.imshow("bg_x",bg)
      # cv.imshow("bg_y",bg_y)
      xx = cv.cvtColor(x, cv.COLOR_GRAY2RGB)
      img = np.concatenate([xx,bg_y,bg], axis=1)
      return img

      # fig, ax  = plt.subplots()
      # ax.imshow(img)
      # ax.set_title("image; ground truth; model output")
      # plt.show()
      # cv.waitKey(-1)

      # fig.savefig(os.path.join(result_path, save_name))
