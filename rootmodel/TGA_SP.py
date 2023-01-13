from __future__ import absolute_import
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision
from torch.nn import init
import numpy as np
class SPM(nn.Module):
    def __init__(self, input_size=64, hidden_size=64):
        super(SPM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Add_Gates = nn.Conv2d(input_size + hidden_size, 2 * hidden_size, 3, padding=3 // 2)
        self.Dot_Gates = nn.Conv2d(input_size + hidden_size, hidden_size, 3, 1, 1)
        self.Out_Gates = nn.Conv2d(input_size + hidden_size, hidden_size, 3, 1, 1)
        self.out_conv = nn.Conv2d(hidden_size, input_size, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, input_, prev_hidden, prev_cell):
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        # generate empty prev_state, if None is provided
        if prev_hidden is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_hidden = Variable(torch.zeros(state_size))
            prev_cell = Variable(torch.zeros(state_size))

        # print(prev_hidden.size())

        prev_hidden = prev_hidden.cuda().detach()
        prev_cell = prev_cell.cuda().detach()

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        Add_gates = self.Add_Gates(stacked_inputs)
        remember_gate = torch.sigmoid(self.Dot_Gates(stacked_inputs))
        out = self.lrelu(self.Out_Gates(stacked_inputs))

        # chunk across channel dimension
        in_gate, cell_gate = Add_gates.chunk(2, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        cell_gate = torch.tanh(cell_gate)

        cell = prev_cell + (in_gate * cell_gate)
        # compute current cell and hidden state
        hidden = out * torch.tanh(cell)
        cell = remember_gate * cell
        # cell = cell + self.lrelu(self.enhence(prev_cell))
        out_hidden = self.lrelu(self.out_conv(hidden))
        return out_hidden, hidden, cell

class Fusion_head(nn.Module):
    def __init__(self, nf, ng):
        super(Fusion_head, self).__init__()
        pad = (0, 1, 1)
        self.conv3d_1 = nn.Conv3d(nf, nf, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv3d_1.apply(initialize_weights)
        self.bn3d_2 = nn.BatchNorm3d(nf, eps=1e-3, momentum=1e-3)
        self.conv3d_2 = nn.Conv3d(nf, ng, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv3d_2.apply(initialize_weights)
        self.bn3d_3 = nn.BatchNorm3d(nf + ng, eps=1e-3, momentum=1e-3)
        self.conv3d_3 = nn.Conv3d(nf + ng, nf + ng, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv3d_3.apply(initialize_weights)
        self.bn3d_4 = nn.BatchNorm3d(nf + ng, eps=1e-3, momentum=1e-3)
        self.conv3d_4 = nn.Conv3d(nf + ng, ng, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv3d_4.apply(initialize_weights)

    def forward(self, x):
        x1 = self.conv3d_1(F.relu(x))
        x1 = self.conv3d_2(F.relu(self.bn3d_2(x1)))
        x1 = torch.cat((x, x1), 1)
        x2 = self.conv3d_3(F.relu(self.bn3d_3(x1)))
        x2 = self.conv3d_4(F.relu(self.bn3d_4(x2)))
        x2 = torch.cat((x1, x2), 1)
        return x2


class Fusion_last(nn.Module):
    def __init__(self, nf, ng):
        super(Fusion_last, self).__init__()
        pad = (0, 1, 1)
        self.bn3d_1 = nn.BatchNorm3d(nf, eps=1e-3, momentum=1e-3)
        self.conv3d_1 = nn.Conv3d(nf, nf, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv3d_1.apply(initialize_weights)
        self.bn3d_2 = nn.BatchNorm3d(nf, eps=1e-3, momentum=1e-3)
        self.conv3d_2 = nn.Conv3d(nf, ng, (3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv3d_2.apply(initialize_weights)
        self.bn3d_3 = nn.BatchNorm3d(nf + ng, eps=1e-3, momentum=1e-3)
        self.conv3d_3 = nn.Conv3d(nf + ng, nf + ng, (1, 1, 1), stride=(1, 1, 1))
        self.conv3d_3.apply(initialize_weights)
        self.bn3d_4 = nn.BatchNorm3d(nf + ng, eps=1e-3, momentum=1e-3)
        self.conv3d_4 = nn.Conv3d(nf + ng, ng, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv3d_4.apply(initialize_weights)

    def forward(self, x):
        x1 = self.conv3d_1(F.relu(self.bn3d_1(x)))
        x1 = self.conv3d_2(F.relu(self.bn3d_2(x1)))
        x1 = torch.cat((x[:, :, 1:-1, :, :], x1), 1)
        x2 = self.conv3d_3(F.relu(self.bn3d_3(x1)))
        x2 = self.conv3d_4(F.relu(self.bn3d_4(x2)))
        x2 = torch.cat((x1, x2), 1)
        return x2


class head(nn.Module):
    def     __init__(self, nf, ng):
        super(head, self).__init__()
        pad = (0, 1, 1)
        self.bn3d_1 = nn.BatchNorm3d(nf, eps=1e-3, momentum=1e-3)
        self.conv3d_1 = nn.Conv3d(nf, nf, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv3d_1.apply(initialize_weights)
        self.bn3d_2 = nn.BatchNorm3d(nf, eps=1e-3, momentum=1e-3)
        self.conv3d_2 = nn.Conv3d(nf, ng, (1, 3, 3), stride=(1, 1, 1), padding=pad)
        self.conv3d_2.apply(initialize_weights)
        self.bn3d_3 = nn.BatchNorm3d(nf + ng, eps=1e-3, momentum=1e-3)
        self.conv3d_3 = nn.Conv3d(nf + ng, nf + ng, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv3d_3.apply(initialize_weights)
        self.bn3d_4 = nn.BatchNorm3d(nf + ng, eps=1e-3)
        self.conv3d_4 = nn.Conv3d(nf + ng, ng, (1, 3, 3), stride=(1, 1, 1), padding=pad)
        self.conv3d_4.apply(initialize_weights)
        self.bn3d_5 = nn.BatchNorm3d(nf + 2 * ng, eps=1e-3, momentum=1e-3)
        self.conv3d_5 = nn.Conv3d(nf + 2 * ng, nf + 2 * ng, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv3d_5.apply(initialize_weights)
        self.bn3d_6 = nn.BatchNorm3d(nf + 2 * ng, eps=1e-3, momentum=1e-3)
        self.conv3d_6 = nn.Conv3d(nf + 2 * ng, ng, (1, 3, 3), stride=(1, 1, 1), padding=pad)
        self.conv3d_6.apply(initialize_weights)

    def forward(self, x):
        x1 = self.conv3d_1(F.relu(self.bn3d_1(x)))
        x1 = self.conv3d_2(F.relu(self.bn3d_2(x1)))
        x1 = torch.cat((x, x1), 1)

        x2 = self.conv3d_3(F.relu(self.bn3d_3(x1)))
        x2 = self.conv3d_4(F.relu(self.bn3d_4(x2)))
        x2 = torch.cat((x1, x2), 1)

        x3 = self.conv3d_5(F.relu(self.bn3d_5(x2)))
        x3 = self.conv3d_6(F.relu(self.bn3d_6(x3)))
        x3 = torch.cat((x2, x3), 1)
        return x3


class middle(nn.Module):
    def __init__(self, nf, ng):
        super(middle, self).__init__()
        pad = (0, 1, 1)
        self.bn3d_1 = nn.BatchNorm3d(nf, eps=1e-3, momentum=1e-3)
        self.conv3d_1 = nn.Conv3d(nf, nf, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv3d_1.apply(initialize_weights)
        self.bn3d_2 = nn.BatchNorm3d(nf, eps=1e-3, momentum=1e-3)
        self.conv3d_2 = nn.Conv3d(nf, ng, (1, 3, 3), stride=(1, 1, 1), padding=pad)
        self.conv3d_2.apply(initialize_weights)
        self.bn3d_3 = nn.BatchNorm3d(nf + ng, eps=1e-3, momentum=1e-3)
        self.conv3d_3 = nn.Conv3d(nf + ng, nf + ng, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv3d_3.apply(initialize_weights)
        self.bn3d_4 = nn.BatchNorm3d(nf + ng, eps=1e-3)
        self.conv3d_4 = nn.Conv3d(nf + ng, ng, (1, 3, 3), stride=(1, 1, 1), padding=pad)
        self.conv3d_4.apply(initialize_weights)
        self.bn3d_5 = nn.BatchNorm3d(nf + 2 * ng, eps=1e-3, momentum=1e-3)
        self.conv3d_5 = nn.Conv3d(nf + 2 * ng, nf + 2 * ng, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv3d_5.apply(initialize_weights)
        self.bn3d_6 = nn.BatchNorm3d(nf + 2 * ng, eps=1e-3, momentum=1e-3)
        self.conv3d_6 = nn.Conv3d(nf + 2 * ng, ng, (1, 3, 3), stride=(1, 1, 1), padding=pad)
        self.conv3d_6.apply(initialize_weights)

    def forward(self, x):
        x1 = self.conv3d_1(F.relu(self.bn3d_1(x)))
        x1 = self.conv3d_2(F.relu(self.bn3d_2(x1)))
        x1 = torch.cat((x, x1), 1)

        x2 = self.conv3d_3(F.relu(self.bn3d_3(x1)))
        x2 = self.conv3d_4(F.relu(self.bn3d_4(x2)))
        x2 = torch.cat((x1, x2), 1)

        x3 = self.conv3d_5(F.relu(self.bn3d_5(x2)))
        x3 = self.conv3d_6(F.relu(self.bn3d_6(x3)))
        x3 = torch.cat((x2, x3), 1)
        return x3


class last(nn.Module):
    def __init__(self, nf, ng):
        super(last, self).__init__()
        pad = (0, 1, 1)
        self.bn3d_1 = nn.BatchNorm3d(nf, eps=1e-3, momentum=1e-3)
        self.conv3d_1 = nn.Conv3d(nf, nf, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv3d_1.apply(initialize_weights)
        self.bn3d_2 = nn.BatchNorm3d(nf, eps=1e-3, momentum=1e-3)
        self.conv3d_2 = nn.Conv3d(nf, ng, (1, 3, 3), stride=(1, 1, 1), padding=pad)
        self.conv3d_2.apply(initialize_weights)
        self.bn3d_3 = nn.BatchNorm3d(nf + ng, eps=1e-3, momentum=1e-3)
        self.conv3d_3 = nn.Conv3d(nf + ng, nf + ng, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv3d_3.apply(initialize_weights)
        self.bn3d_4 = nn.BatchNorm3d(nf + ng, eps=1e-3)
        self.conv3d_4 = nn.Conv3d(nf + ng, ng, (1, 3, 3), stride=(1, 1, 1), padding=pad)
        self.conv3d_4.apply(initialize_weights)
        self.bn3d_5 = nn.BatchNorm3d(nf + 2 * ng, eps=1e-3, momentum=1e-3)
        self.conv3d_5 = nn.Conv3d(nf + 2 * ng, nf + 2 * ng, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv3d_5.apply(initialize_weights)
        self.bn3d_6 = nn.BatchNorm3d(nf + 2 * ng, eps=1e-3, momentum=1e-3)
        self.conv3d_6 = nn.Conv3d(nf + 2 * ng, ng + 1, (1, 3, 3), stride=(1, 1, 1), padding=pad)
        self.conv3d_6.apply(initialize_weights)
        self.ng = ng
        self.SPM = SPM(input_size=ng * 3)
        self.fusion = nn.Conv2d(ng*3*2, ng*3, 3, 1, 1)

    def forward(self, x, last_h, last_c):
        x1 = self.conv3d_1(F.relu(self.bn3d_1(x)))
        x1 = self.conv3d_2(F.relu(self.bn3d_2(x1)))
        x1 = torch.cat((x, x1), 1)

        x2 = self.conv3d_3(F.relu(self.bn3d_3(x1)))
        x2 = self.conv3d_4(F.relu(self.bn3d_4(x2)))
        x2 = torch.cat((x1, x2), 1)

        x3 = self.conv3d_5(F.relu(self.bn3d_5(x2)))
        x3 = self.conv3d_6(F.relu(self.bn3d_6(x3)))
        x_att = x3[:, self.ng:self.ng + 1, :, :, :]

        x_att = F.softmax(x_att, dim=2)
        x_feat = x3[:, :self.ng, :, :, :]
        b, t, c, h, w = x_feat.size()
        x_SPM_in = x_feat.view(b, -1, h, w)
        x_SPM_out, next_h, next_c = self.SPM(x_SPM_in, last_h, last_c)
        x_fusion_out = self.fusion(torch.cat((x_SPM_in, x_SPM_out), dim=1))
        x_feat = x_fusion_out.view(b, t, c, h, w)
        x3 = torch.mul(x_att, x_feat)
        x3 = torch.cat((x2, x3), 1)
        return x3, x_att, next_h, next_c


class TGA(nn.Module):
    def __init__(self, scale=4):
        super(TGA, self).__init__()
        self.scale = scale
        nf = 64
        ng = 16
        pad = (0, 1, 1)
        self.conv3d_1 = nn.Conv3d(3, nf, (1, 3, 3), stride=(1, 1, 1), padding=pad)
        self.conv3d_1.apply(initialize_weights)
        self.bn3d_1 = nn.BatchNorm3d(nf, eps=1e-3, momentum=1e-3)
        self.conv3d_2 = nn.Conv3d(nf, nf, (1, 3, 3), stride=(1, 1, 1), padding=pad)
        self.conv3d_2.apply(initialize_weights)
        self.bn3d_2 = nn.BatchNorm3d(nf, eps=1e-3, momentum=1e-3)
        self.conv3d_2_1 = nn.Conv3d(nf, nf, (1, 3, 3), stride=(1, 1, 1), padding=pad)
        self.conv3d_2_1.apply(initialize_weights)
        self.bn3d_2_1 = nn.BatchNorm3d(nf, eps=1e-3, momentum=1e-3)
        self.conv3d_2_2 = nn.Conv3d(nf, nf, (3, 3, 3), stride=(3, 1, 1), padding=pad)
        self.conv3d_2_2.apply(initialize_weights)
        self.head = head(nf, ng)
        self.middle_1 = middle(nf + 3 * ng, ng)
        self.middle_2 = middle(nf + 6 * ng, ng)
        self.middle_3 = middle(nf + 9 * ng, ng)
        self.middle_4 = middle(nf + 12 * ng, ng)
        self.last = last(nf + 15 * ng, ng)
        self.fusion_head = Fusion_head(nf + 18 * ng, ng)
        self.Fusion_last = Fusion_last(nf + 20 * ng, ng)
        self.bn3d_2_2 = nn.BatchNorm3d(nf + 22 * ng, eps=1e-3, momentum=1e-3)
        self.compress = nn.Conv3d(nf + 22 * ng, nf, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.compress.apply(initialize_weights)
        self.middle_6 = middle(nf, ng)
        self.middle_7 = middle(nf + 3 * ng, ng)
        self.middle_8 = middle(nf + 6 * ng, ng)
        self.middle_9 = middle(nf + 9 * ng, ng)
        self.middle_10 = middle(nf + 12 * ng, ng)
        self.middle_11 = middle(nf + 15 * ng, ng)
        self.middle_12 = middle(nf + 18 * ng, ng)
        self.bn3d_3 = nn.BatchNorm3d(nf + 21 * ng, eps=1e-3, momentum=1e-3)
        self.conv3d_3 = nn.Conv3d(nf + 21 * ng, 256, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv3d_3.apply(initialize_weights)
        self.conv3d_r1 = nn.Conv3d(256, 256, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv3d_r1.apply(initialize_weights)
        self.conv3d_r2 = nn.Conv3d(256, 128 * 4, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv3d_r2.apply(initialize_weights)

        self.conv3d_r4 = nn.Conv3d(128, 128, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv3d_r4.apply(initialize_weights)

        self.conv3d_r3 = nn.Conv3d(128, scale * 3, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv3d_r3.apply(initialize_weights)

    def forward(self, x, last_h, last_c):
        x_center = x[:,1,:,:,:].contiguous()
        x = x.permute(0, 2, 1, 3, 4)
        B, C, T, H, W = x.shape

        x = F.relu(self.bn3d_1(self.conv3d_1(x)))
        x = F.relu(self.bn3d_2(self.conv3d_2(x)))
        x = F.relu(self.bn3d_2_1(self.conv3d_2_1(x)))
        x = self.conv3d_2_2(x)
        x = self.head(x)
        x = self.middle_1(x)
        x = self.middle_2(x)
        x = self.middle_3(x)
        x = self.middle_4(x)
        if last_h is None:
            x, x_att, next_h, next_c = self.last(x, None, None)
        else:
            x, x_att, next_h, next_c = self.last(x, last_h, last_c)
        x = self.fusion_head(x)
        x = self.Fusion_last(x)
        x = self.compress(F.relu(self.bn3d_2_2(x)))
        x = self.middle_6(x)
        x = self.middle_7(x)
        x = self.middle_8(x)
        x = self.middle_9(x)
        x = self.middle_10(x)
        x = self.middle_11(x)
        x = self.middle_12(x)
        x = F.relu(self.conv3d_3(F.relu(self.bn3d_3(x))))
        Rx = F.relu(self.conv3d_r1(x))
        Rx = self.conv3d_r2(Rx)
        Rx = F.relu(F.pixel_shuffle(Rx.squeeze_(2), 2))
        Rx = torch.unsqueeze(Rx, dim=2)
        Rx = F.relu(self.conv3d_r4(Rx))
        Rx = self.conv3d_r3(Rx)
        base = F.interpolate(x_center, scale_factor=4, mode='bilinear', align_corners=False)
        out = base + F.pixel_shuffle(Rx.squeeze_(2), 2)
        return out, next_h, next_c


def initialize_weights(net_l, scale=0.1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()