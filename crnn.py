import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import ipdb


class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut, drop=0):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, dropout=drop) # dropout=0.3
        self.embedding = nn.Linear(nHidden * 2, nOut) # *2: because of bidirectional

    def forward(self, input):
        # ipdb.set_trace()
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output

class crnn(nn.Module):
    def __init__(self, class_dim, channel, multi):
        super(crnn, self).__init__()

        self.class_dim = class_dim
        self.in_channels = channel
        self.cnn = nn.Sequential()

        def convRelu(cnn, i, channel_in, channel_out, kernel, pad, strd, batchNormal=False):
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(in_channels=channel_in,
                                     out_channels=channel_out,
                                     kernel_size=kernel,
                                     padding=pad,
                                     stride=strd,
                                     bias=False))
            if batchNormal:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(channel_out))
            cnn.add_module('relu{0}'.format(i), nn.ReLU(inplace=True))

        # 0
        convRelu(self.cnn, 0, self.in_channels, 64, 3, 1, 1, batchNormal=False)
        self.cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, stride=2))

        # 1
        convRelu(self.cnn, 1, 64, 128, 3, 1, 1, batchNormal=False)
        self.cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, stride=2))

        # 2
        convRelu(self.cnn, 2, 128, 256, 3, 1, 1, batchNormal=True)

        # 3
        convRelu(self.cnn, 3, 256, 256, 3, 1, 1, batchNormal=False)
        self.cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d((2, 2), (2, 1), (0, 1))) # kernel, stride, padding (長, 高)

        # 4
        convRelu(self.cnn, 4, 256, 512, 3, 1, 1, batchNormal=True)

        # 5
        convRelu(self.cnn, 5, 512, 512, 3, 1, 1, batchNormal=False)
        self.cnn.add_module('pooling{0}'.format(5), nn.MaxPool2d((2, 2), (2, 1), (0, 1))) # kernel, stride, padding (長, 高)

        # 6
        convRelu(self.cnn, 6, 512, 512, 1, 0, 1, batchNormal=True)

        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, 256, 256),
            BidirectionalLSTM(256, 256, self.class_dim))  # 0~9 -. /_ (10+4+1)

    def forward(self, input_img):
        # conv features
        conv = self.cnn(input_img)
        batch, channel, height, width = conv.size()
        assert height == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        
        # rnn features
        output = self.rnn(conv)
        output = nn.functional.log_softmax(output, 2)
        # output = output.view(b, -1, 15)

        return output
        