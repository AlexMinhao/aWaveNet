import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.preprocessing import StandardScaler
import math
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np
import pywt
NB_SENSOR_CHANNELS = 9
# Hardcoded number of classes in the gesture recognition problem
NUM_CLASSES = 6
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
qBASE_lr = 0.0001
EPOCH = 100


class DeepConvLSTM(nn.Module):
    def __init__(self):
        super(DeepConvLSTM, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=NUM_FILTERS, kernel_size=(1, FILTER_SIZE)),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, 2*NUM_FILTERS, kernel_size=(1, FILTER_SIZE)),
            nn.BatchNorm2d(2*NUM_FILTERS),
            nn.Dropout2d(0.2),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(2*NUM_FILTERS, NUM_FILTERS, kernel_size=(1, FILTER_SIZE)),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.Dropout2d(0.2),
            nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, 1, kernel_size=(1, FILTER_SIZE)),
            # nn.BatchNorm2d(NUM_FILTERS),
            # nn.Dropout2d(0.5),
            nn.ReLU())
        self.conv5 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=(1, FILTER_SIZE)),
            # nn.BatchNorm2d(NUM_FILTERS),
            # nn.Dropout2d(0.5),
            nn.ReLU())
        self.lstm = nn.LSTM(9, NUM_UNITS_LSTM, NUM_LSTM_LAYERS, batch_first=True)

        self.fc = nn.Linear(NUM_UNITS_LSTM, NUM_CLASSES)

    def forward(self, x):
        # print (x.shape)
        b, l, c = x.size()
        x = x.view(b, 1, c, l)
        out = self.conv1(x)
        # print (out.shape)
        out = self.conv2(out)
        # print (out.shape)
        out = self.conv3(out)
        # print (out.shape)
        out = self.conv4(out)
        # print (out.shape)
        # out = out.view(-1, NB_SENSOR_CHANNELS, NUM_FILTERS)

        out = out.view(-1, b, 9)  # CHANNELS_NUM_50

        h0 = Variable(torch.zeros(NUM_LSTM_LAYERS, out.size(0), NUM_UNITS_LSTM))
        c0 = Variable(torch.zeros(NUM_LSTM_LAYERS, out.size(0), NUM_UNITS_LSTM))
        if torch.cuda.is_available():
            h0, c0 = h0.cuda(), c0.cuda()

        # forward propagate rnn


        out, _ = self.lstm(out, (h0, c0))
        out = out.permute(1, 0, 2)
        features = out[:, -1, :]
        #  out[:, -1, :] -> [100,11,128] ->[100,128]
        out = self.fc(out[:, -1, :])
        return out,[0]


config_info = {
    'data_folder': 'G://Research//NewAction//Dataset//UCI_HAR_Dataset//',
    'data_folder_raw': 'G://Research//NewAction//Dataset//UCI_HAR_Dataset//',
    'result_file': 'result_har.csv',
    'epoch': 500,
    'lr': 0.003,
    'batch_size': 32,
    'momemtum': 0.9
}

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
result = []
BASE_lr = config_info['lr']

def adjust_learning_rate(optimizer,  epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    lr = BASE_lr * (0.5 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_variable(x):
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x


# This is for parsing the X data, you can ignore it if you do not need preprocessing
def format_data_x(datafile):
    x_data = None
    for item in datafile:
        item_data = np.loadtxt(item, dtype=np.float)
        if x_data is None:
            x_data = np.zeros((len(item_data), 1))
        x_data = np.hstack((x_data, item_data))
    x_data = x_data[:, 1:]
    print(x_data.shape)
    X = None
    for i in range(len(x_data)):
        row = np.asarray(x_data[i, :])
        row = row.reshape(9, 128).T
        if X is None:
            X = np.zeros((len(x_data), 128, 9))
        X[i] = row
    print(X.shape)
    return X


# This is for parsing the Y data, you can ignore it if you do not need preprocessing
def format_data_y(datafile):
    data = np.loadtxt(datafile, dtype=np.int) - 1
    YY = np.eye(6)[data]
    return YY


# Load data function, if there exists parsed data file, then use it
# If not, parse the original dataset from scratch
def load_data():
    import os
    if os.path.isfile(config_info['data_folder'] + '\data_har.npz') == True:
        data = np.load(config_info['data_folder'] + 'data_har.npz')
        X_train = data['X_train']
        Y_train = data['Y_train']
        X_test = data['X_test']
        Y_test = data['Y_test']
    else:
        # This for processing the dataset from scratch
        # After downloading the dataset, put it to somewhere that str_folder can find
        str_folder = config_info['data_folder_raw']
        INPUT_SIGNAL_TYPES = [
            "body_acc_x_",
            "body_acc_y_",
            "body_acc_z_",
            "body_gyro_x_",
            "body_gyro_y_",
            "body_gyro_z_",
            "total_acc_x_",
            "total_acc_y_",
            "total_acc_z_"
        ]

        str_train_files = [str_folder + '\\train' + '\\Inertial_Signals\\' + item + 'train.txt' for item in
                           INPUT_SIGNAL_TYPES]
        str_test_files = [str_folder + '\\test' + '\\Inertial_Signals\\' +
                          item + 'test.txt' for item in INPUT_SIGNAL_TYPES]
        str_train_y = str_folder + '\\train\\y_train.txt'
        str_test_y = str_folder + '\\test\\y_test.txt'

        X_train = format_data_x(str_train_files)
        X_test = format_data_x(str_test_files)
        Y_train = format_data_y(str_train_y)
        Y_test = format_data_y(str_test_y)

    return X_train, onehot_to_label(Y_train), X_test, onehot_to_label(Y_test)


def onehot_to_label(y_onehot):
    a = np.argwhere(y_onehot == 1)
    return a[:, -1]


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


def normalize(x):
    x_min = x.min(axis=(0, 2, 3), keepdims=True)
    x_max = x.max(axis=(0, 2, 3), keepdims=True)
    x_norm = (x - x_min) / (x_max - x_min)
    return x_norm


def load(batch_size=64):
    x_train, y_train, x_test, y_test = load_data()

    transform = None
    train_set = data_loader(x_train, y_train, transform)
    test_set = data_loader(x_test, y_test, transform)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def normalization(data):
    for i in range(data.shape[0]):
        _range = np.max(data[i]) - np.min(data[i])
        data[i] = (data[i] - np.min(data[i])) / _range
    return data


def standardization(data):
    for i in range(data.shape[0]):
        mu = np.mean(data[i], axis=0)
        sigma = np.std(data[i], axis=0)
        data[i] = (data[i] - mu) / sigma
    return data



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
        self.modified = modified

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
            # x_odd = self.ptemp(x_odd)
            # x_odd =self.U(x_odd) #18 65
            c = x_even + self.U(x_odd)
            d = x_odd - self.P(c) #         Todo: +  -> * -> sigmod

            # c = x_even + self.seNet_P(x_odd)
            # d = x_odd - self.seNet_P(c)
            return (c, d)
        else:

            x_even = x_even.permute(0, 2, 1)
            x_odd = x_odd.permute(0, 2, 1)
            # a = self.phi(x_even)
            d = x_odd.mul(torch.exp(self.phi(x_even))) - self.P(x_even)
            c = x_even.mul(torch.exp(self.psi(d))) + self.U(d)
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
    def __init__(self, in_planes, out_planes):
        super(BottleneckBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.relu = nn.ReLU(inplace=True)

        self.disable_conv = in_planes == out_planes
        if not self.disable_conv:
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

            self.bootleneck = BottleneckBlock(in_planes * 1, in_planes * 1)
        else:
            self.bootleneck = BottleneckBlock(in_planes * 4, in_planes * 2)

    def forward(self, x):
        (L, H) = self.wavelet(x) #10 9 128
        x = L
        details = H

        r = None
        if(self.regu_approx + self.regu_details != 0.0):  #regu_details=0.01, regu_approx=0.01

            if self.regu_details:
                rd = self.regu_details * \
                     H.abs().mean()


            # Constrain on the approximation
            if self.regu_approx:
                rc = self.regu_approx * torch.dist(L.mean(), x.mean(), p=2)


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
            return self.bootleneck(x).permute(0, 2, 1), r, details
        else:
            return x, r, details

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
            self.bootleneck = BottleneckBlock(in_planes * 1, in_planes * 1)
        else:
            self.bootleneck = BottleneckBlock(in_planes * 4, in_planes * 2)

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
                 lifting_size=[2, 1], kernel_size=4, no_bootleneck=False,
                 classifier="mode2", share_weights=False, simple_lifting=False,
                  regu_details=0.01, regu_approx=0.01, haar_wavelet=False):
        super(WASN, self).__init__()

        self.initialization = False
        self.nb_channels_in = first_conv
        self.level_part = number_level_part
        # First convolution
        if first_conv != 3 and first_conv != 9 and first_conv != 22 :
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
            bootleneck = True
            if no_bootleneck and i == number_levels - 1:
                bootleneck = False
            if i == 0:

                if haar_wavelet:
                    self.levels.add_module(
                        'level_' + str(i),
                        Haar(in_planes,
                             lifting_size, kernel_size, bootleneck,
                             share_weights, simple_lifting, regu_details, regu_approx)
                    )
                else:
                    self.levels.add_module(
                        'level_' + str(i),
                        LevelWASN(in_planes,
                                  lifting_size, kernel_size, bootleneck,
                                  share_weights, simple_lifting, regu_details, regu_approx)
                    )
            else:
                if haar_wavelet:
                    self.levels.add_module(
                        'level_' + str(i),
                        Haar(in_planes,
                             lifting_size, kernel_size, bootleneck,
                             share_weights, simple_lifting, regu_details, regu_approx)
                    )
                else:
                    self.levels.add_module(
                        'level_' + str(i),
                        LevelWASN(in_planes,
                                  lifting_size, kernel_size, bootleneck,
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






def train(model, optimizer, train_loader, test_loader):
    n_batch = len(train_loader.dataset) // config_info['batch_size']
    criterion = nn.CrossEntropyLoss()
    test_pred = np.empty((0))
    test_true = np.empty((0))
    train_pred = np.empty((0))
    train_true = np.empty((0))
    for e in range(config_info['epoch']):
        model.train()
        adjust_learning_rate(optimizer, e)
        correct, total_loss = 0, 0
        total = 0
        for index, (sample, target) in enumerate(train_loader):
            #             sample, target = sample.float(), target.long()
            # sample = sample.view(-1, 9, 1, 128)
            seqs, target = sample.to(
                DEVICE).float(), target.to(DEVICE).long()
            # sample = sample.view(-1, 9, 1, 128)
            output, regus = model(seqs)
            loss_class = criterion(output, target)
            loss_total = loss_class
            # If no regularisation used, None inside regus
            if regus[0]:
                loss_regu = sum(regus)
                loss_total += loss_regu
                is_regu_activated = True

            loss = loss_total

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum()
            train_pred = np.append(train_pred, predicted.cpu().numpy(), axis=0)
            train_true = np.append(train_true, target.cpu().numpy(), axis=0)

            if index % 20 == 0:
                print(
                    'Epoch: [{}/{}], Batch: [{}/{}], loss:{:.4f},lr: {}'.format(e + 1, config_info['epoch'], index + 1,
                                                                                n_batch,
                                                                                loss.item(),
                                                                                optimizer.param_groups[0]['lr']))
        macroF1_each = [f1_score(train_true, train_pred, labels=[i], average='macro') for i in
                        range(6)]
        F1_macro_train = f1_score(train_true, train_pred, average='macro')

        acc_train = float(correct) * 100.0 / \
                    (config_info['batch_size'] * n_batch)
        print(
            'Epoch: [{}/{}], loss: {:.4f}, train acc: {:.2f}%, train f1: {:.2f}%'.format(e + 1, config_info['epoch'],
                                                                                         total_loss * 1.0 / n_batch,
                                                                                         acc_train,
                                                                                         F1_macro_train * 100))

        # Testing
        model.train(False)
        with torch.no_grad():
            correct, total = 0, 0
            for (sample, target) in test_loader:
                #                 sample, target = sample.float(), target.long()
                seqs, target = sample.to(
                    DEVICE).float(), target.to(DEVICE).long()
                # sample = sample.view(-1, 9, 1, 128)
                output, regus = model(seqs)
                loss_class = criterion(output, target)
                loss_total = loss_class
                # If no regularisation used, None inside regus
                if regus[0]:
                    loss_regu = sum(regus)
                    loss_total += loss_regu
                    is_regu_activated = True

                loss = loss_total
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum()

                test_pred = np.append(test_pred, predicted.cpu().numpy(), axis=0)
                test_true = np.append(test_true, target.cpu().numpy(), axis=0)

        macroF1_each = [f1_score(test_true, test_pred, labels=[i], average='macro') for i in
                        range(6)]
        F1_macro = f1_score(test_true, test_pred, average='macro')
        F1_weighted = f1_score(test_true, test_pred, average='weighted')
        acc_test = float(correct) * 100 / total
        print(macroF1_each)
        print('Epoch: [{}/{}], test acc: {:.2f}% f1_macro:{:.2f}%'.format(e + 1,
                                                                          config_info['epoch'],
                                                                          float(correct) * 100 / total, F1_macro * 100))
        result.append([acc_train, acc_test, F1_macro * 100, F1_weighted*100, loss.item()])
        result_np = np.array(result, dtype=float)
        np.savetxt('UCI_results/DeepConvLstm_adj0.003_150.csv', result_np, fmt='%.2f', delimiter=',')

def plot():
    data = np.loadtxt('result.csv', delimiter=',')
    plt.figure()
    plt.plot(range(1, len(data[:, 0]) + 1),
             data[:, 0], color='blue', label='train')
    plt.plot(range(1, len(data[:, 1]) + 1),
             data[:, 1], color='red', label='test')
    plt.plot(range(1, len(data[:, 2]) + 1),
             data[:, 2], color='green', label='test_f1')
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Training and Test Accuracy', fontsize=20)
    plt.show()




torch.manual_seed(10)
train_loader, test_loader = load(
    batch_size=config_info['batch_size'])
model = WASN(num_classes=6, first_conv=9,
                 number_levels=9,
                 number_level_part=[[1, 1], [1, 1], [0, 0], [1, 1], [0, 0], [1, 1], [0,0], [0, 0], [0, 0]], haar_wavelet=False).to(DEVICE)

# model = DeepConvLSTM().to(DEVICE)
optimizer = optim.SGD(params=model.parameters(
), lr=config_info['lr'], momentum=config_info['momemtum'],weight_decay=0.01)
train(model, optimizer, train_loader, test_loader)
# result = np.array(result, dtype=float)
# np.savetxt(config_info['result_file'], result, fmt='%.2f', delimiter=',')
plot()
