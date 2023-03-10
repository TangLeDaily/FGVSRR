import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms
from torch.utils.data import DataLoader

from util import *

def load_img(filepath):
    img = Image.open(filepath)
    return img

def get_pic_name_lis(rootpath, path): #datasets/train/   and   000/
    imgls = []
    for i in os.listdir(rootpath + "input/" + path):
        imgls.append(i)
    imglss = sorted(imgls)
    return imglss

def get_path(rootpath): #datasets/train/
    pathls = []
    for i in os.listdir(rootpath + "input/"):
        pathls.append(i+"/")
    return pathls

def get_total_path(rootpath, mid, pathls):
    out = []
    for path in pathls:
        out.append(rootpath + mid + path)
    return out

def rand():
    return random.random()
class RandomCrop(object):
    def __init__(self):
        self.scale = 4
        self.output_sizeX = 64
        self.output_sizeY = 64
    def __call__(self, sample):
        lrC, lrN, hr, ranintX, ranintY = sample['lrC'], sample['lrN'], sample['hr'], sample['ranintX'], sample['ranintY']
        h, w = lrC.shape[1],lrC.shape[2]
        new_h, new_w = self.output_sizeX, self.output_sizeY
        toplr = int((h - new_h) * ranintX)
        leftlr = int((w - new_w) * ranintY)
        newlrC = lrC[:, toplr:toplr + new_h, leftlr: leftlr + new_w]
        newlrN = []
        for i in range(len(lrN)):
            newlrN.append(lrN[i][:, toplr:toplr + new_h, leftlr: leftlr + new_w])

        hh, ww = hr.shape[1], hr.shape[2]
        new_hh, new_ww = self.output_sizeX * self.scale, self.output_sizeY * self.scale
        tophr = toplr*self.scale
        lefthr = leftlr*self.scale
        newhr = hr[:, tophr:tophr + new_hh, lefthr: lefthr + new_ww]

        return newlrC, newlrN, newhr

class train_data_set(data.Dataset):
    def __init__(self, rootpath, batchsize): #datasets/train/
        super(train_data_set, self).__init__()
        self.batchsize = batchsize
        self.pathls = get_path(rootpath) # 000/ 001/ 002/
        self.input_path_ls = get_total_path(rootpath, "input/", self.pathls)  #datasets/train/input/000/
        self.target_path_ls = get_total_path(rootpath, "target/", self.pathls)
        self.pic_name_lis = get_pic_name_lis(rootpath, self.pathls[0])
        self.transLR = transforms.Compose([
            transforms.ToTensor()
            ])
        self.transHR = transforms.Compose([
            transforms.ToTensor()
            ])

        self.lastID = []
        self.randintX = []
        self.randintY = []
        for i in range(len(self.input_path_ls) + 200):
            self.randintX.append(rand())
            self.randintY.append(rand())
        self.crop = transforms.Compose([RandomCrop()])
    def __len__(self):

        return len(self.input_path_ls) * len(self.pic_name_lis)

    def __getitem__(self, idx):
        video_Bo = idx // (self.batchsize * len(self.pic_name_lis))  # 16 // 8 * 100 = 0,  821 // 8 * 100 = 1
        video_ID = idx % self.batchsize  # 16 % 8 = 0, 821 % 8 = 5
        video_real_ID = self.batchsize * video_Bo + video_ID  # 0*8 + 0 = 0,  1*8 + 5 = 13
        frame = idx % (self.batchsize * len(
            self.pic_name_lis)) // self.batchsize  # 16 % 800 // 8 = 2, 821 % 800 // 8 = 21 // 8 = 2
        if video_real_ID > 265:
            video_real_ID = video_real_ID - 265
        lrC = self.transLR(load_img(self.input_path_ls[video_real_ID] + self.pic_name_lis[frame]))
        lrN = []
        if frame < 2:
            for i in range(5):
                if i != frame:
                    lrN.append(self.transLR(load_img(self.input_path_ls[video_real_ID] + self.pic_name_lis[i])))
        elif frame > len(self.pic_name_lis) - 3:
            for i_b in range(5):
                if len(self.pic_name_lis) - i_b - 1 != frame:
                    lrN.append(self.transLR(load_img(
                        self.input_path_ls[video_real_ID] + self.pic_name_lis[len(self.pic_name_lis) - i_b - 1])))
        else:
            for i_r in range(2):
                lrN.append(
                    self.transLR(load_img(self.input_path_ls[video_real_ID] + self.pic_name_lis[frame - i_r - 1])))
                lrN.append(
                    self.transLR(load_img(self.input_path_ls[video_real_ID] + self.pic_name_lis[frame + i_r + 1])))
        hr = self.transHR(load_img(self.target_path_ls[video_real_ID] + self.pic_name_lis[frame]))
        sample = {'lrC': lrC, 'lrN': lrN, 'hr': hr, 'ranintX': self.randintX[video_real_ID], 'ranintY': self.randintY[video_real_ID]}
        lrCC, lrNN, hrr = self.crop(sample)
        # lrCC, lrNN, hrr = lrC, lrN, hr
        lr_final = []
        lr_final.append(lrCC)
        for i in range(4):
            lr_final.append(lrNN[i])
        lrr = torch.stack(lr_final, dim=0)
        return lrr, hrr

class test_data_set(data.Dataset):
    def __init__(self, rootpath, path): #datasets/test/  and 000/
        super(test_data_set, self).__init__()
        self.feat_ID = "test"
        self.input_path = rootpath + "input/" + path  #datasets/test/input/000/
        self.target_path = rootpath + "target/" + path
        self.pic_name_lis = get_pic_name_lis(rootpath, path)
        self.transLR = transforms.Compose([
            transforms.ToTensor()])
        self.transHR = transforms.Compose([
            transforms.ToTensor()])

    def __len__(self):
        return len(self.pic_name_lis)

    def __getitem__(self, idx):
        lrc = self.transLR(load_img(self.input_path + self.pic_name_lis[idx]))
        lrN = []
        frame = idx
        if frame < 2:
            for i in range(5):
                if i != frame:
                    lrN.append(self.transLR(load_img(self.input_path + self.pic_name_lis[i])))
        elif frame > len(self.pic_name_lis) - 3:
            for i_b in range(5):
                if len(self.pic_name_lis) - i_b - 1 != frame:
                    lrN.append(self.transLR(load_img(
                        self.input_path + self.pic_name_lis[len(self.pic_name_lis) - i_b - 1])))
        else:
            for i_r in range(2):
                lrN.append(
                    self.transLR(load_img(self.input_path + self.pic_name_lis[frame - i_r - 1])))
                lrN.append(
                    self.transLR(load_img(self.input_path + self.pic_name_lis[frame + i_r + 1])))
        hr = self.transHR(load_img(self.target_path + self.pic_name_lis[idx]))
        lr_final  = []
        lr_final.append(lrc)
        for i in range(4):
            lr_final.append(lrN[i])
        lr = torch.stack(lr_final, dim=0)
        return lr, hr

if __name__ == "__main__":
    dataset = train_data_set("datasets/train/", 16)
    training_data_loader = DataLoader(dataset=dataset, batch_size=1, num_workers=0,
                                      drop_last=True)
    for i, batch in enumerate(training_data_loader,1):
        lr, hr = batch
        if i % 100 == 0:
            save_pic(lr[0, 0, :, :, :], "data/test/pic{}/".format(i), "lr5_{}.png".format(i))
            save_pic(lr[0, 1, :, :, :], "data/test/pic{}/".format(i), "lr5_neb_1_{}.png".format(i))
            save_pic(lr[0, 2, :, :, :], "data/test/pic{}/".format(i), "lr5_neb_2_{}.png".format(i))
            save_pic(lr[0, 3, :, :, :], "data/test/pic{}/".format(i), "lr5_neb_3_{}.png".format(i))
            save_pic(lr[0, 4, :, :, :], "data/test/pic{}/".format(i), "lr5_neb_4_{}.png".format(i))
            save_pic(hr[0, :, :, :], "data/test/pic{}/".format(i), "hr5_{}.png".format(i))