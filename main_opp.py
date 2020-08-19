import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as cp
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast
from torch.utils.data import Dataset, DataLoader
from torch import optim

from sklearn.metrics import f1_score
from time import time
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix

import json
from numpy.lib.stride_tricks import as_strided as ast
from utils import SeqGenerator, distance, LongShortdistance
from torch.utils.data import DataLoader

from torch import nn
from torch import optim
import torch
# from utils import Logger, data_augmentation, pre_emphasis
import os

from sklearn.metrics import f1_score
import joblib
from torch.utils.data import Dataset, DataLoader
import pickle as cp
import pywt
NB_SENSOR_CHANNELS = 113
# Hardcoded number of classes in the gesture recognition problem
NUM_CLASSES = 18
# Hardcoded length of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_LENGTH = 48
# Hardcoded step of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_STEP = 12
# Length of the input sequence after convolutional operations
FINAL_SEQUENCE_LENGTH = 8
# Batch Size
BATCH_SIZE = 64
# Number filters convolutional layers
NUM_FILTERS = 64
# Size filters convolutional layers
FILTER_SIZE = 5
# Number of unit in the long short-term recurrent layers
NUM_UNITS_LSTM = 128
NUM_LSTM_LAYERS = 2
BASE_lr = 0.0003
EPOCH = 100



def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if pretrain:
        # if epoch < 20:
        #     lr = 0.0000005
        # else:
        lr = 0.000005
        # lr = 0.0000003
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        # Decay learning rate exponentially
        # if epoch < 5:
        #     lr = 0.00000598 * epoch + 0.0000001
        # else:
        #     lr = 0.5 * (1 + math.cos(3.14 / EPOCH * epoch)) * 0.00003
        #
        # #lr = BASE_lr * (args.lr_decay)**epoch
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr

        lr = BASE_lr * (0.5 ** (epoch // 20))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def load_opp_dataset_official(path, enhence , ws, ss):
    #path = 'C:\ALEX\Doc\Reference\PytorchTuto\OPPORTUNITY\oppChallenge_gestures.data'
    f = open(path, 'rb')
    data = cp.load(f)
    f.close()

    X_train, y_train = data[0]
    # X_s4_train, y_s4_train = data[2]
    # X_train = np.concatenate((X_train, X_s4_train), axis=0)
    # y_train = np.concatenate((y_train, y_s4_train), axis=0)
    X_test, y_test = data[1]

    print(" ..from file {}".format(path))
    print(" ..reading instances: train {0}, test {1}".format(X_train.shape, X_test.shape))

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # The targets are casted to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    return X_train.astype(np.float32),  y_train.astype(np.uint8), X_test.astype(np.float32), y_test.astype(np.uint8)




def load_imu_data(path):

    with open(path, 'r') as load_f:
        load_dict = json.load(load_f)

    return load_dict


def norm_shape(shape):
    '''
    Normalize numpy array shapes so they're always expressed as a tuple,
    even for one-dimensional shapes.

    Parameters
        shape - an int, or a tuple of ints

    Returns
        a shape tuple
    '''
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass

    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass

    raise TypeError('shape must be an int, or a tuple of ints')

def sliding_window(a,ws,ss = None,flatten = True):
    '''
    Return a sliding window over a in any number of dimensions

    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an
                  extra dimension for each dimension of the input.

    Returns
        an array containing each n-dimensional window from a
    '''

    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)

    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)


    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape), len(ws), len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError(\
        'ws cannot be larger than a in any dimension.\
 a.shape was %s and ws was %s' % (str(a.shape),str(ws)))

    # how many slices will there be in each dimension?
    newshape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a,shape = newshape,strides = newstrides)
    if not flatten:
        return strided

    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    dim = list(filter(lambda i : i != 1,dim))
    return strided.reshape(dim)




class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()
        # Deciding the stride base on the direction

        self.conv_even = lambda x: x[:, ::2, :]
        self.conv_odd = lambda x: x[:, 1::2, :]


    def forward(self, x):
        '''Returns the odd and even part'''
        return (self.conv_even(x), self.conv_odd(x))


class LiftingScheme(nn.Module):
    def __init__(self, in_planes, modified=False, size=[], splitting=True, k_size=4, dropout = 0.5, simple_lifting=False):
        super(LiftingScheme, self).__init__()
        self.modified = True #modified

        kernel_size = k_size
        pad = k_size // 2 # 2 1 0 0

        self.splitting = splitting
        self.split = Splitting()

        # Dynamic build sequential network
        modules_P = []
        modules_U = []
        modules_psi = []
        modules_phi = []
        prev_size = 1

        # HARD CODED Architecture
        if simple_lifting:
            modules_P += [
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_planes, in_planes,
                          kernel_size=kernel_size, stride=1),
                nn.Tanh()
            ]
            modules_U += [
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_planes, in_planes,
                          kernel_size=kernel_size, stride=1),
                nn.Tanh()
            ]
        else:
            size_hidden = 4                          # Todo:   extend

            modules_P += [
                nn.ReflectionPad1d(pad),
                # nn.ReplicationPad1d(pad),
                nn.Conv1d(in_planes * prev_size, in_planes * size_hidden,   #3-6
                          kernel_size=kernel_size, stride=1),
                                                           #  Todo:  BN
                nn.ReLU(),
                # nn.BatchNorm1d(in_planes * size_hidden, momentum=0.1),
                # nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(dropout),                           # Todo: change value
            ]
            modules_U += [
                nn.ReplicationPad1d(pad),
                # nn.ReplicationPad1d(pad),
                nn.Conv1d(in_planes * prev_size, in_planes * size_hidden,
                          kernel_size=kernel_size, stride=1),
                nn.ReLU(),                              # Todo: leakyrelu  0.1
                # nn.BatchNorm1d(in_planes * size_hidden, momentum=0.1),
                # nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(dropout),
            ]
            prev_size = size_hidden

            # Final dense
            modules_P += [
                nn.Conv1d(in_planes * prev_size, in_planes,
                          kernel_size=2, stride=1),
                nn.Tanh(),                                 # Todo: leakyrelu  0.1
                # nn.BatchNorm1d(in_planes , momentum=0.1),
                # nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(dropout),
            ]
            modules_U += [
                nn.Conv1d(in_planes * prev_size, in_planes,            #6-3
                          kernel_size=2, stride=1),
                nn.Tanh(),
                # nn.BatchNorm1d(in_planes, momentum=0.1),
                # nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(dropout),
            ]

            modules_phi += [
                nn.ReplicationPad1d(pad),
                nn.Conv1d(in_planes, in_planes,
                          kernel_size=kernel_size, stride=1),
                nn.Conv1d(in_planes, in_planes,
                          kernel_size=2, stride=1),
                nn.Tanh()
            ]
            modules_psi += [
                nn.ReplicationPad1d(pad),
                nn.Conv1d(in_planes, in_planes,
                          kernel_size=kernel_size, stride=1),
                nn.Conv1d(in_planes, in_planes,
                          kernel_size=2, stride=1),
                nn.Tanh()
            ]

        self.P = nn.Sequential(*modules_P)
        self.U = nn.Sequential(*modules_U)
        self.phi = nn.Sequential(*modules_phi)
        self.psi = nn.Sequential(*modules_psi)


    def forward(self, x):
        if self.splitting:
            #3  224  112
            #3  112  112
            (x_even, x_odd) = self.split(x)
        else:
            (x_even, x_odd) = x

        if self.modified:
            x_even = x_even.permute(0, 2, 1)
            x_odd = x_odd.permute(0, 2, 1)
            # a = self.phi(x_even)
            d = x_odd.mul(torch.exp(self.phi(x_even))) - self.P(x_even)
            c = x_even.mul(torch.exp(self.psi(d))) + self.U(d)
            return (c, d)

        else:

            x_even = x_even.permute(0, 2, 1)
            x_odd = x_odd.permute(0, 2, 1)
            # x_odd = self.ptemp(x_odd)
            # x_odd =self.U(x_odd) #18 65
            c = x_even + self.U(x_odd)
            d = x_odd - self.P(c)  # Todo: +  -> * -> sigmod

            # c = x_even + self.seNet_P(x_odd)
            # d = x_odd - self.seNet_P(c)
            return (c, d)


class LiftingSchemeLevel(nn.Module):
    def __init__(self, in_planes, share_weights, modified=False, size=[2, 1], kernel_size=4, simple_lifting=False):
        super(LiftingSchemeLevel, self).__init__()
        self.level = LiftingScheme(
             in_planes=in_planes, modified=modified,
            size=size, k_size=kernel_size, simple_lifting=simple_lifting)


    def forward(self, x):
        '''Returns (LL, LH, HL, HH)'''
        (L, H) = self.level(x)  #10 3 224 224

        return (L, H)



class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, disable_conv):
        super(BottleneckBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.relu = nn.ReLU(inplace=True)

        self.disable_conv = disable_conv#in_planes == out_planes
        # if not self.disable_conv:
        self.conv1 = nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)

    def forward(self, x):
        if self.disable_conv:
            return self.relu(self.bn1(x))
        else:
            return self.conv1(self.relu(self.bn1(x)))

class LevelWASN(nn.Module):
    def __init__(self, in_planes, lifting_size, kernel_size, no_bottleneck,
                 share_weights, simple_lifting, regu_details, regu_approx):
        super(LevelWASN, self).__init__()
        self.regu_details = regu_details
        self.regu_approx = regu_approx
        if self.regu_approx + self.regu_details > 0.0:

            self.loss_details = nn.SmoothL1Loss()

        self.wavelet = LiftingSchemeLevel(in_planes, share_weights,
                                       size=lifting_size, kernel_size=kernel_size,
                                       simple_lifting=simple_lifting)
        self.share_weights = share_weights
        if no_bottleneck:
            # We still want to do a BN and RELU, but we will not perform a conv
            # as the input_plane and output_plare are the same
            self.bootleneck = BottleneckBlock(in_planes, in_planes, disable_conv = False)
        else:
            self.bootleneck = BottleneckBlock(in_planes, in_planes, disable_conv = True)

    def forward(self, x):
        (L, H) = self.wavelet(x) #10 9 128
        approx = L
        details = H

        r = None
        if(self.regu_approx + self.regu_details != 0.0):  #regu_details=0.01, regu_approx=0.01

            if self.regu_details:
                rd = self.regu_details * \
                     details.abs().mean()


            # Constrain on the approximation
            if self.regu_approx:
                rc = self.regu_approx * torch.dist(approx.mean(), x.mean(), p=2)


            if self.regu_approx == 0.0:
                # Only the details
                r = rd
            elif self.regu_details == 0.0:
                # Only the approximation
                r = rc
            else:
                # Both
                r = rd + rc

        if self.bootleneck:
            return self.bootleneck(approx).permute(0, 2, 1), r, details
        else:
            return approx, r, details

class Haar(nn.Module):
    def __init__(self, in_planes, lifting_size, kernel_size, no_bottleneck,
                 share_weights, simple_lifting, regu_details, regu_approx):
        super(Haar, self).__init__()
        from pytorch_wavelets import DWTForward

        self.regu_details = regu_details
        self.regu_approx = regu_approx
        # self.wavelet = pywt.dwt([1, 2, 3, 4, 5, 6], 'db1')#DWTForward(J=1, mode='zero', wave='db1').cuda()
        self.share_weights = share_weights
        if no_bottleneck:
            # We still want to do a BN and RELU, but we will not perform a conv
            # as the input_plane and output_plare are the same
            self.bootleneck = BottleneckBlock(in_planes, in_planes, disable_conv = False)
        else:
            self.bootleneck = BottleneckBlock(in_planes, in_planes, disable_conv = True)

    def forward(self, x):
        input = x.permute(0, 2, 1)
        input = input.cpu().detach().numpy()

        L, H = pywt.dwt(input, 'db1') #self.wavelet(x)
        approx = get_variable(torch.from_numpy(L))
        details = get_variable(torch.from_numpy(H))
        approx = approx.permute(0, 2, 1)
        # details = details.permute(0, 2, 1)
        # LH = H[0][:, :, 0, :, :]
        # HL = H[0][:, :, 1, :, :]
        # HH = H[0][:, :, 2, :, :]
        #
        # x = LL
        # details = torch.cat([LH, HL, HH], 1)
        r = None
        if (self.regu_approx + self.regu_details != 0.0):
            # Constraint on the details
            if self.regu_details:
                rd = self.regu_details * \
                     details.abs().mean()

            # Constrain on the approximation
            if self.regu_approx:
                rc = self.regu_approx * torch.dist(approx.mean(), x.mean(), p=2)

            if self.regu_approx == 0.0:
                # Only the details
                r = rd
            elif self.regu_details == 0.0:
                # Only the approximation
                r = rc
            else:
                # Both
                r = rd + rc

        return approx, r, details


class WASN(nn.Module):
    def __init__(self, num_classes, big_input=True, first_conv=9, extend_channel = 128,
                 number_levels=4, number_level_part=[[1, 0], [1, 0], [1, 0]],
                 lifting_size=[2, 1], kernel_size=4, no_bootleneck=True,
                 classifier="mode2", share_weights=False, simple_lifting=False,
                  regu_details=0.01, regu_approx=0.01, haar_wavelet=False):
        super(WASN, self).__init__()

        self.initialization = False
        self.nb_channels_in = first_conv
        self.level_part = number_level_part
        # First convolution
        if first_conv != 3 and first_conv != 9 and first_conv != 77 :
            self.first_conv = True
            self.conv1 = nn.Sequential(
                nn.Conv1d(first_conv, extend_channel,
                          kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm1d(extend_channel),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Conv1d(extend_channel, extend_channel,
                          kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm1d(extend_channel),
                nn.ReLU(True),
                nn.Dropout(0.5),
            )
            in_planes = extend_channel
            out_planes = extend_channel * (number_levels + 1)
        else:
            self.first_conv = False
            in_planes = first_conv
            out_planes = first_conv * (number_levels + 1)


        self.levels = nn.ModuleList()


        for i in range(number_levels):
            # bootleneck = True
            # if no_bootleneck and i == number_levels - 1:
            #     bootleneck = False
            if i == 0:

                if haar_wavelet:
                    self.levels.add_module(
                        'level_' + str(i),
                        Haar(in_planes,
                             lifting_size, kernel_size, no_bootleneck,
                             share_weights, simple_lifting, regu_details, regu_approx)
                    )
                else:
                    self.levels.add_module(
                        'level_' + str(i),
                        LevelWASN(in_planes,
                                  lifting_size, kernel_size, no_bootleneck,
                                  share_weights, simple_lifting, regu_details, regu_approx)
                    )
            else:
                if haar_wavelet:
                    self.levels.add_module(
                        'level_' + str(i),
                        Haar(in_planes,
                             lifting_size, kernel_size, no_bootleneck,
                             share_weights, simple_lifting, regu_details, regu_approx)
                    )
                else:
                    self.levels.add_module(
                        'level_' + str(i),
                        LevelWASN(in_planes,
                                  lifting_size, kernel_size, no_bootleneck,
                                  share_weights, simple_lifting, regu_details, regu_approx)
                    )
            in_planes *= 1


            out_planes += in_planes * 3

        if no_bootleneck:
            in_planes *= 1

        self.num_planes = out_planes


        if classifier == "mode1":
            self.fc = nn.Linear(out_planes, num_classes)
        elif classifier == "mode2":

            self.fc = nn.Sequential(
                nn.Linear(in_planes*(number_levels + 1), 1024),  # Todo:  extend channels
                nn.BatchNorm1d(1024),
                nn.ReLU(True),
                nn.Linear(1024, num_classes)
            )
        else:
            raise "Unknown classifier"

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight.data)
                # if m.bias is not None:
                m.bias.data.zero_()
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.atten = MultiHeadAttention(n_head=5, d_model=in_planes, d_k=64, d_v=64, dropout=0.3)
        self.count_levels = 0
    def forward(self, x):

        # Todo : add bn
        if self.first_conv:
            x = x.permute(0, 2, 1)
            x = self.conv1(x)
            x = x.permute(0, 2, 1)

        rs = []  # List of constrains on details and mean
        det = []  # List of averaged pooled details

        input = [x, ]
        for l in self.levels:
            low, r, details = l(input[0])
            if self.level_part[self.count_levels][0]:
                input.append(low)
            else:
                low = low.permute(0, 2, 1)
                det += [self.avgpool(low)]
            if self.level_part[self.count_levels][1]:
                details = details.permute(0, 2, 1)
                input.append(details)
            else:
                det += [self.avgpool(details)]
            del input[0]
            rs += [r]
            self.count_levels = self.count_levels + 1

        for aprox in input:
            aprox = aprox.permute(0, 2, 1)  # b 77 1
            aprox = self.avgpool(aprox)
            det += [aprox]

        self.count_levels = 0
        # We add them inside the all GAP detail coefficients
        x = torch.cat(det, 2)
        x = x.permute(0, 2, 1)
        q, att = self.atten(x, x, x, mask=None)
        x = q
        b, c, l = x.size()
        x = x.view(-1, c * l)
        #
        # det += [aprox]
        # x = torch.cat(det, 2)
        # b, c, l = x.size()
        # x = x.view(-1, c * l)

        return self.fc(x), rs

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

                        # 512    8  64    64
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn



class data_loader(Dataset):
    def __init__(self, samples, labels, t):
        self.samples = samples
        self.labels = labels
        self.T = t

    def __getitem__(self, index):
        sample, target = self.samples[index], self.labels[index]
        if self.T:
            return self.T(sample), target
        else:
            return sample, target

    def __len__(self):
        return len(self.samples)


def get_sample_weights(y, weights):
    '''
    to assign weights to each sample
    '''
    label_unique = np.unique(y)
    sample_weights = []
    for val in y:
        idx = np.where(label_unique == val)
        sample_weights.append(weights[idx])
    return sample_weights

def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)


def checkpoint(epoch, model, optimizer):
    model_out_path = os.path.join(os.getcwd(), r'results', "model_best_adjust.pth")
    states = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(states, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.3):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class FeatureExtractor(nn.Module):

    def __init__(self, submodule):

        super(FeatureExtractor, self).__init__()

        self.submodule = submodule

        self.avgpool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):

        outputs = []
        rs = []  # List of constrains on details and mean
        det = []  # List of averaged pooled details
        for name, module in self.submodule._modules.items():
            if name is "levels":
                for l in module:
                    x, r, details = l(x)  # b  77 L
                    # Add the constrain of this level
                    rs += [r]
                    # Globally avgpool all the details
                    det += [self.avgpool(details)]  # b 77 1

                x = x.permute(0, 2, 1)  # b 77 1
                aprox = self.avgpool(x)
                # We add them inside the all GAP detail coefficients

                # det += [aprox]
                L = torch.squeeze(aprox)
                H1 = torch.squeeze(det[0])
                H2 = torch.squeeze(det[1])
                H3 = torch.squeeze(det[2])
                H4 = torch.squeeze(det[3])
                # x = torch.cat(det, 1)
                # x = x.view(-1, x.size()[1])


        return L, H1, H2, H3, H4


def get_variable(x):
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x

is_dwnn = 1

def train_epoch_action(epoch, train_loader, model, loss_function, optimizer,f1_train_epoch_weighted,f1_train_epoch_macro, result):
    print('train at epoch {}'.format(epoch + 1))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    train_pred = np.empty((0))
    train_true = np.empty((0))

    end_time = time()
    adjust_learning_rate(optimizer, epoch)

    for i, (seqs, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time() - end_time)

        seqs = get_variable(seqs.float()) #[Batch,1,window,113]
        labels = get_variable(labels.long()) #[B]

        if is_dwnn:
            output, regus = model(seqs)
            loss_class = loss_function(output, labels)
            loss_total = loss_class
            # If no regularisation used, None inside regus
            if regus[0]:
                loss_regu = sum(regus)
                loss_total += loss_regu
                is_regu_activated = True
            loss = loss_total
            losses.update(loss)
        else:
            output, regus = model(seqs)
            loss = loss_function(output, labels)
            losses.update(loss)

        _, preds = torch.max(output.data, 1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



        f1_train_weighted =  f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='weighted')
        f1_train_macro = f1_score(labels.cpu().numpy(), preds.cpu().numpy(),  average='macro')

        batch_time.update(time() - end_time)
        end_time = time()

        train_pred = np.append(train_pred, preds.cpu().numpy(),axis=0)
        train_true = np.append(train_true, labels.cpu().numpy(),axis=0)

        if (i + 1) % 10 == 0:
            print(
                'Each Train_Iter [%d/%d] Loss: %.6f, Loss_avg: %.6f, F1-score_weighted: %.3f, F1-score_macro: %.3f'
                % (i + 1, len(train_loader), loss.item(), losses.avg, f1_train_weighted, f1_train_macro))




    f1_train_epoch_weighted.update(f1_score(train_true, train_pred, average='weighted'))
    f1_train_epoch_macro.update(f1_score(train_true, train_pred, average='macro'))
    macroF1_each = [f1_score(train_true, train_pred, labels=[i], average='macro') for i in range(18)]
    re = np.concatenate((macroF1_each, [f1_train_epoch_weighted.val], [f1_train_epoch_macro.val], [losses.avg], [optimizer.param_groups[0]['lr']]))
    result.append(re)

    confusion = confusion_matrix(train_true, train_pred, labels = Labels)
    confusion_np = np.array(confusion, dtype=float)
    # np.savetxt('result_train_confusion_77_adjust_bt64_ramdom_lr999.csv', confusion_np, fmt='%.4f', delimiter=',')

    print(
        'Final: Epoch [%d/%d], Loss: %.6f,  Time: %.3f, F1-score_weighted.avg: %.3f,'
        ' F1-score_macro.avg: %.3f,lr: %.7f '
        % (epoch + 1, EPOCH, losses.avg,
           batch_time.val, f1_train_epoch_weighted.avg, f1_train_epoch_macro.avg, optimizer.param_groups[0]['lr']))
    print(
        'Epoch Each class f1 macro Train_Iter', macroF1_each)

# In[20]:


def val_epoch_action(epoch, valition_loader, model, optimizer, loss_function, f1_epoch_test_weighted, f1_epoch_test_macro, result):
    print('validation at epoch {}'.format(epoch + 1))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end_time = time()
    test_pred = np.empty((0))
    test_true = np.empty((0))
    event_acc, event_miF, event_maF, frame_acc, frame_miF, frame_maF = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    predicted_label_segment, lengths_varying_segment, true_label_segment = torch.LongTensor(), torch.LongTensor(), torch.LongTensor()

    with torch.no_grad():
        for i, (seqs, labels) in enumerate(valition_loader):
            # measure data loading time
            data_time.update(time() - end_time)

            seqs = get_variable(seqs.float())
            labels = get_variable(labels.long())

            if is_dwnn:
                output, regus = model(seqs)
                loss_class = loss_function(output, labels)
                loss_total = loss_class
                # If no regularisation used, None inside regus
                if regus[0]:
                    loss_regu = sum(regus)
                    loss_total += loss_regu
                    is_regu_activated = True
                loss = loss_total
                losses.update(loss)
            else:
                output, regus = model(seqs)
                loss = loss_function(output, labels)
                losses.update(loss)

            labels = labels.squeeze()
            _, preds = torch.max(output.data, 1)

            # losses.update(loss.data, seqs.size(0))


            f1_test_weighted = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='weighted')
            f1_test_macro = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')

            batch_time.update(time() - end_time)
            end_time = time()
            test_pred = np.append(test_pred, preds.cpu().numpy(), axis=0)
            test_true = np.append(test_true, labels.cpu().numpy(), axis=0)

            predicted_label_segment = torch.LongTensor(torch.cat((predicted_label_segment, preds.cpu()), dim=0))

            true_label_segment = torch.LongTensor(torch.cat((true_label_segment, labels.cpu()), dim=0))

            if (i + 1) % 10 == 0:
                print(
                    'Validation_Iter [%d/%d], F1-weighted-score: %.3f, F1-macro-score: %.3f'
                    % (i + 1, len(valition_loader), f1_test_weighted, f1_test_macro))


        print(
            'Epoch: [{}/{}], e acc:{:.2f}%, e_miF:{:.3f}%, e maF:{:.3f}%'.format(
                epoch + 1, EPOCH, event_acc, event_miF, event_maF))


        macroF1_each = [f1_score(test_true, test_pred, labels=[i], average='macro') for i in
                        range(18)]
        f1_epoch_test_weighted.update(f1_score(test_true, test_pred, average='weighted'))
        f1_epoch_test_macro.update(f1_score(test_true, test_pred, average='macro'))

        re = np.concatenate((macroF1_each, [f1_epoch_test_weighted.val], [f1_epoch_test_macro.val], [losses.avg],[optimizer.param_groups[0]['lr']] ))
        result.append(re)

        confusion = confusion_matrix(test_true, test_pred, labels=Labels)
        confusion_np = np.array(confusion, dtype=float)
        # np.savetxt('result_test_confusion_77_adjust_bt64_ramdom_lr999.csv', confusion_np, fmt='%.4f', delimiter=',')
        print("TestTrueSize: {0}".format(test_true.shape))
        print("TestPredSize: {0}".format(test_pred.shape))
        print('Final Epoch [%d/%d], Time: %.3f, F1-score_weighted.avg: %.5f, F1-score_macro.avg: %.5f'
                 % (epoch + 1, EPOCH, batch_time.val, f1_score(test_true, test_pred, average='weighted'), f1_score(test_true, test_pred, average='macro')))

        print(
            'Each class f1 macro Train_Iter',macroF1_each)
        # Save checkpoint if necessary

        if epoch % 5 == 0:
            chk_path = os.path.join('checkpoint/Best_adj_epoch_{}.bin'.format(epoch))
            print('Saving checkpoint to', chk_path)
            torch.save({
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, chk_path)
            print('Model has been saved as:',chk_path)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    torch.manual_seed(4321)  # reproducible
    np.random.seed(4321)
    torch.cuda.manual_seed_all(4321)
    DEVICE = torch.device('cuda:0')

    Labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    is_dwnn = 1
    pretrain =0
    result_train = []
    result_test = []
    print("Loading data...")
    # path = "C:\\ALEX\\Doc\\Reference\\SoundNet\\Dataset\\oppChallenge_gestures.data"
    path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                        r'Dataset\oppChallenge_gestures.data')
    X_train, y_train, X_test, y_test = load_opp_dataset_official(path, 12, SLIDING_WINDOW_LENGTH,
                                                                 SLIDING_WINDOW_STEP)  # load_dataset(path) #load_opp_dataset_official(path, SLIDING_WINDOW_LENGTH,
    # SLIDING_WINDOW_STEP)  # load_dataset(path)

    model = WASN(num_classes=18, first_conv=77,
                 number_levels=4).cuda()  # TemporalModel(in_features = 45, out_features=NUM_CLASSES,
    # filter_widths=filter_widths, causal=True, dropout=args.dropout, channels=args.channels).cuda()#Baseline(2160, 2, NUM_CLASSES).cuda()
    model = WASN(num_classes=18, first_conv=77,
                 number_levels=7,
                 number_level_part=[[1, 1], [1, 1], [0, 0], [1, 1], [0, 0], [0, 0], [0, 0]],
                 haar_wavelet=False).to(DEVICE)

    print(model)
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
    print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    for k, v in model.state_dict().items():
        print(k, v.shape)
    loss_function = LabelSmoothing()  # nn.CrossEntropyLoss()#BalancedCrossEntropyLoss(weight =weights) #nn.CrossEntropyLoss()LabelSmoothing()#
    optimizer = optim.Adam(model.parameters(), lr=BASE_lr, weight_decay=1e-4, amsgrad=True)

    if pretrain:
        pre_train_path = os.path.join(os.getcwd(),
                                      r'checkpoint\Best77_4_1024_r_s_9192_7371.bin')
        pretrain_model = torch.load(pre_train_path)
        model.load_state_dict(pretrain_model['state_dict'])
        optimizer.load_state_dict(pretrain_model['optimizer'])
        print(model)

    for epoch in range(EPOCH):
        # In[11]:
        # Sensor data is segmented using a sliding window mechanism
        if SLIDING_WINDOW_LENGTH == 1:
            X_test1, y_test1 = X_test, y_test
            X_train1, y_train1 = X_train, y_train
        else:
            X_test1, y_test1 = opp_sliding_window(X_test, y_test, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
            X_train1, y_train1 = opp_sliding_window(X_train[epoch:], y_train[epoch:], SLIDING_WINDOW_LENGTH,
                                                    SLIDING_WINDOW_STEP)
            print(" ..after sliding window (testing): inputs {0}, targets {1}".format(X_test1.shape, y_test1.shape))
            print(" ..after sliding window (train): inputs {0}, targets {1}".format(X_train1.shape, y_train1.shape))
            X_test1 = X_test1.reshape((-1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))
            X_train1 = X_train1.reshape((-1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))
        # Data is reshaped since the input of the network is a 4 dimension tensor


        X_test1 = X_test1[:, :, 36:113]
        X_train1 = X_train1[:, :, 36:113]
        # X_test1 = X_test
        # X_train1 = X_train

        X_train1 = list(X_train1)
        y_train1 = list(y_train1)
        X_test1 = list(X_test1)
        y_test1 = list(y_test1)
        training_set = [(X_train1[i], y_train1[i]) for i in range(len(y_train1))]
        testing_set = [(X_test1[i], y_test1[i]) for i in range(len(y_test1))]


        train_loader = DataLoader(dataset=training_set, batch_size=BATCH_SIZE, shuffle=True)  # , sampler = sampler
        test_loader = DataLoader(dataset=testing_set, batch_size=BATCH_SIZE, shuffle=False)


        f1_train_weighted = AverageMeter()
        f1_train_macro = AverageMeter()
        f1_test_weighted = AverageMeter()
        f1_test_macro = AverageMeter()
        train_epoch_action(epoch, train_loader, model, loss_function, optimizer, f1_train_weighted, f1_train_macro,
                           result_train)
        val_epoch_action(epoch, test_loader, model, optimizer, loss_function, f1_test_weighted, f1_test_macro,
                         result_test)

        result_train_np = np.array(result_train, dtype=float)
        np.savetxt('OPP_results/result_train_77_4_1024_win48_0.00003_atten5_64_111100110000_adj_modified_100epochOpp.csv', result_train_np, fmt='%.7f',
                   delimiter=',')

        result_test_np = np.array(result_test, dtype=float)
        np.savetxt('OPP_results/result_test_77_4_1024_win48_0.00003_atten5_64__111100110000_adj_modified_100epochOpp.csv', result_test_np, fmt='%.7f', delimiter=',')
