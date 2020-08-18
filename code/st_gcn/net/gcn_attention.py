import torch
import torch.nn as nn
from torch.autograd import Variable
from .net import conv_init
from .spatial_transformer import spatial_attention
import numpy as np

'''Unit that implements augmented graph spatial convolution in case of Spatial Transformer
in_channels: number of channels in input in the convolutional unit
(for the first one is 3 because it corresponds to the number of coordinates)
out_channels: number of channels in output (for the last one is 256)
incidence: incidence matrix of the entire skeleton
incidence is a PxVxV matrix where p is the number of partitions for that specific partition strategy
(see paper for further details) '''

scale_norm = False


class gcn_unit_attention(nn.Module):
    def __init__(self, in_channels, out_channels, incidence, num, dv_factor, dk_factor, Nh, complete, relative,
                 only_attention, layer, more_channels, drop_connect, data_normalization, skip_conn, adjacency, padding=0,
                 kernel_size=1,
                 stride=1, bn_flag=True,
                 t_dilation=1, last_graph=False, visualization=True):
        super().__init__()
        self.incidence = incidence
        self.incidence = incidence
        self.relu = nn.ReLU()
        self.visualization=visualization
        self.in_channels = in_channels
        self.more_channels = more_channels
        self.drop_connect = drop_connect
        self.data_normalization=data_normalization
        self.skip_conn=skip_conn
        self.adjacency = adjacency
        print("Nh ", Nh)
        print("Dv ", dv_factor)
        print("Dk ", dk_factor)

        self.last_graph = last_graph
        if (not only_attention):
            self.out_channels = out_channels - int((out_channels) * dv_factor)
        else:
            self.out_channels = out_channels
        self.data_bn = nn.BatchNorm1d(self.in_channels * 25)
        self.bn = nn.BatchNorm2d(out_channels)
        self.only_attention = only_attention
        self.bn_flag = bn_flag
        self.layer = layer

        self.incidence = Variable(self.incidence.clone(), requires_grad=False).view(-1, self.incidence.size()[-1],
                                                                                    self.incidence.size()[-1])

        # Each Conv2d unit implements 2d convolution to weight every single partition (filter size 1x1)
        # There is a convolutional unit for each partition
        # This is done only in the case in which Spatial Transformer and Graph Convolution are concatenated

        if (not self.only_attention):
            self.g_convolutions = nn.ModuleList(

                [nn.Conv2d(in_channels, self.out_channels, kernel_size=(kernel_size, 1), padding=(padding, 0),
                           stride=(stride, 1), dilation=(t_dilation, 1)) for i in
                 range(self.incidence.size()[0])]
            )
            for conv in self.g_convolutions:
                conv_init(conv)

            self.attention_conv = spatial_attention(in_channels=self.in_channels, kernel_size=1,
                                                dk=int(out_channels * dk_factor),
                                                dv=int(out_channels * dv_factor), Nh=Nh, complete=complete,
                                                relative=relative,
                                                stride=stride, layer=self.layer, A=self.incidence, num=num,
                                                more_channels=self.more_channels,
                                                drop_connect=self.drop_connect,
                                                data_normalization=self.data_normalization, skip_conn=self.skip_conn,
                                                adjacency=self.adjacency, visualization=self.visualization)
        else:
            self.attention_conv = spatial_attention(in_channels=self.in_channels, kernel_size=1,
                                                dk=int(out_channels * dk_factor),
                                                dv=int(out_channels), Nh=Nh, complete=complete,
                                                relative=relative,
                                                stride=stride, last_graph=self.last_graph, layer=self.layer,
                                                A=self.incidence, num=num, more_channels=self.more_channels,
                                                drop_connect=self.drop_connect,
                                                data_normalization=self.data_normalization, skip_conn=self.skip_conn,
                                                adjacency=self.adjacency, visualization=self.visualization)


    def forward(self, x, label, name):
        # N: number of samples, equal to the batch size
        # C: number of channels, in our case 3 (coordinates x, y, z)
        # T: number of frames
        # V: number of nodes
        N, C, T, V = x.size()
        x_sum = x
        if (self.data_normalization):
            x = x.permute(0, 1, 3, 2).reshape(N, C * V, T)
            x = self.data_bn(x)
            x = x.reshape(N, C, V, T).permute(0, 1, 3, 2)

        # Adjacence matrix
        self.incidence = self.incidence.cuda(x.get_device())

        # Learnable parameter
        incidence = self.incidence

        # N, T, C, V > NT, C, 1, V
        xa = x.permute(0, 2, 1, 3).reshape(-1, C, 1, V)

        # Another normalization called "ScaleNorm" that we tried on our data
        if scale_norm:
            self.scale = ScaleNorm(scale=C ** 0.5)
            xa = self.scale(xa)

        # Spatial Transformer
        attn_out = self.attention_conv(xa, label, name)
        # N, T, C, V > N, C, T, V
        attn_out = attn_out.reshape(N, T, -1, V).permute(0, 2, 1, 3)

        if (not self.only_attention):

            # For each partition multiplies for the input and applies convolution 1x1 to the result to weight each partition
            for i, partition in enumerate(incidence):
                # print(partition)
                # NCTxV
                xp = x.reshape(-1, V)
                # (NCTxV)*(VxV)
                xp = xp.mm(partition.float())
                # NxCxTxV
                xp = xp.reshape(N, C, T, V)

                if i == 0:
                    y = self.g_convolutions[i](xp)
                else:
                    y = y + self.g_convolutions[i](xp)

            # Concatenate on the channel dimension the two convolutions
            y = torch.cat((y, attn_out), dim=1)
        else:
            if self.skip_conn and self.in_channels == self.out_channels:
                y = attn_out + x_sum
            else:
                y = attn_out
        if (self.bn_flag):
            y = self.bn(y)

        y = self.relu(y)

        return y


class ScaleNorm(nn.Module):
    """ScaleNorm"""

    def __init__(self, scale, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.scale = scale

        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=1, keepdim=True).clamp(min=self.eps)
        return x * norm
