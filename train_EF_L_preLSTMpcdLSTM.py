import argparse
import os
import random

import wandb
import numpy
import torch
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import nn
from PIL import Image, ImageFilter
from torchvision import transforms
from torch.nn import functional as F

from dataset_E import *
from util import *
from model import *
from srresnet import *
from rootmodel.EFVSR_L_preLSTMpcdOneLSTM import *

parser = argparse.ArgumentParser(description="PyTorch Data_Pre")
parser.add_argument("--train_root_path", default='datasets/train/', type=str, help="train root path")
parser.add_argument("--test_root_path", default='datasets/test/REDS4/', type=str, help="test root path")
parser.add_argument("--cuda", default=True, action="store_true", help="use cuda?")
parser.add_argument("--frame", default=100, type=int, help="use cuda?")
parser.add_argument("--model_mark", default=0, type=int, help="which model to train? 0:default")
parser.add_argument("--resume", default='', type=str, help="path to latest checkpoint (default: none)")
parser.add_argument("--start_epoch", default=0, type=int, help="manual epoch number (useful on restarts)")
parser.add_argument("--batchSize", type=int, default=16, help="training batch size")  # default 16
parser.add_argument("--nEpochs", type=int, default=10000, help="number of epochs to train for")
parser.add_argument("--miniEpochs", type=int, default=0, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate. Default=1e-4")
parser.add_argument("--threads", type=int, default=8, help="number of threads for data loader to use")
parser.add_argument("--scale", type=int, default=4, help="Scale default:4x")
parser.add_argument("--loss", type=int, default=0, help="the loss function, default")
use_wandb = True
opt = parser.parse_args()
min_avr_loss = 99999999
save_flag = 0
epoch_avr_loss = 0
n_iter = 0
in_nc = 3
out_nc = 3


def get_yu(model):
    kk = torch.load("checkpoints/EFVSR_L_OnlypcdOneLSTM/model_epoch_623_psnr_27.2809.pth", map_location='cpu')
    torch.save(kk.state_dict(), "checkpoints/state/New_125.pth")
    pretrained_dict = torch.load("checkpoints/state/New_125.pth")
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)  # 利用预训练模型的参数，更新模型
    model.load_state_dict(model_dict)
    return model

def get_yu2(model):
    kk = torch.load("checkpoints/default/net_g_300000.pth")
    model_dict = model.state_dict()
    kk = {k: v for k, v in kk.items() if k in model_dict}
    model_dict.update(kk)  # 利用预训练模型的参数，更新模型
    model.load_state_dict(model_dict)
    return model

def main():
    global model, opt
    str = "None"
    if opt.model_mark == 0:
        str = "EFVSR_L_preLSTMpcdOneLSTM"
    else:
        str = "ERROR"
    if use_wandb:
        wandb.init(project="FGVSR_V3", name=str, entity="karledom")
    print(opt)

    print("===> Find Cuda")
    cuda = opt.cuda
    # torch.cuda.set_device(1)
    torch.cuda.set_device(0)
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    opt.seed = random.randint(1, 10000)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

    print("===> Loading datasets")
    train_set = train_data_set(opt.train_root_path, batchsize=opt.batchSize)
    training_data_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize, num_workers=opt.threads,
                                      drop_last=True)

    print("===> Building model")
    if opt.model_mark == 0:
        model = EFVSR()
    # else:
    #     model = NLVR()
    # 加载模型
    criterion = nn.MSELoss()

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    print("===> Do Resume Or Skip")
    # checkpoint = torch.load("checkpoints/EFVSR_L_OnlypcdOneLSTM/model_epoch_25_psnr_26.4176.pth")
    # model.load_state_dict(checkpoint.state_dict())
    model = get_yu(model)
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.lo = 0
            model.load_state_dict(checkpoint.state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    print("===> Setting Optimizer")
    # for p in model.pcd_align.parameters():
     #    p.requires_grad = False
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(optimizer, model, criterion, epoch, training_data_loader)
        psnr = test_train_set(model, epoch)
        save_checkpoint(model, psnr, epoch)


def del_train_feat(filename):
    for i in range(3):
        if os.path.exists("data/feature/" + filename + "_{}.npy".format(i)):
            os.remove("data/feature/" + filename + "_{}.npy".format(i))


def train(optimizer, model, criterion, epoch, train_dataloader):
    global min_avr_loss
    global save_flag
    global epoch_avr_loss
    global n_iter
    global opt

    avr_loss = 0

    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    # del_train_feat()
    model.train()
    avg_loss = AverageMeter()
    for iteration, batch in enumerate(train_dataloader):
        input, target, feat_I_D = batch
        feat_ID = iteration // opt.frame + 3000
        if iteration % opt.frame == 0:
            del_train_feat(str(feat_ID))
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()
        out = model(input, feat_ID)
        loss = criterion(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss.update(loss.item())
        if iteration % 50 == 0:
            # inp = F.interpolate(input, scale_factor=4, mode='bilinear', align_corners=False)
            # psnrroot = calc_psnr(inp[0,:,:,:], target[0,:,:,:])
            psnrroot = 100
            if use_wandb:
                wandb.log({'epoch': epoch, 'iter_loss': avg_loss.avg, 'psnr_root':psnrroot})
            print('epoch_iter_{}_ID_{}_loss is {:.10f}_psnrroot_is_{:.4f}'.format(iteration, feat_ID, avg_loss.avg,psnrroot))
        if iteration % 500 == 0:
            psnrr = test_train_set(model, epoch)
    psnr = test_train_set(model, epoch)
    save_checkpoint(model, psnr, epoch)


def save_checkpoint(model, psnr, epoch):
    global min_avr_loss
    global save_flag
    global opt

    if opt.model_mark == 0:
        model_folder = "checkpoints/EFVSR_L_preLSTMpcdOneLSTM/"
    else:
        model_folder = "checkpoints/error/"
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    model_out_path = model_folder + "model_epoch_{}_psnr_{:.4f}.pth".format(epoch, psnr)
    # state = {"epoch": epoch, "model": model}
    torch.save(model, model_out_path)
    # torch.save(model.state_dict(), "checkpoints/SPVSR/state/model_epoch_{}_state_psnr_{:.4f}.pth".format(epoch, psnr))
    print("Checkpoint saved to {}".format(model_out_path))

    if save_flag is True:
        torch.save(model, '{}epoch_{}_min_batch_loss_{}.pth'.format(model_folder, epoch, min_avr_loss))

        print('min_loss model saved')


def test_train_set(this_model, epoch_num):
    print(" -- Start eval --")
    del_train_feat("EFVSR_L_OnlypcdOneLSTM")
    test_set = test_data_set(opt.test_root_path, "000/")
    test_loader = DataLoader(dataset=test_set, batch_size=1, num_workers=opt.threads)
    psnr = AverageMeter()
    with torch.no_grad():
        model = this_model
        if opt.cuda:
            model = model.cuda()
        model.eval()
        for iteration, batch in enumerate(test_loader, 1):
            input, target, I_D = batch
            ID = "EFVSR_L_OnlypcdOneLSTM"
            if opt.cuda:
                input = input.cuda()
                target = target.cuda()
            out = model(input, ID)
            psnr.update(calc_psnr(out, target), len(out))
        if use_wandb:
            wandb.log({'epoch': epoch_num, 'psnr': psnr.avg})
        print("--->This--EFVSR_L_preLSTMpcdOneLSTM--epoch:{}--Avg--PSNR: {:.4f} dB--Root--PSNR: 24.11 dB".format(epoch_num,
                                                                                               psnr.avg))
    return psnr.avg


if __name__ == "__main__":
    main()
