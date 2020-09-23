import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import tqdm
from .temporal_transformer_windowed import tcn_unit_attention_block
from .temporal_transformer import tcn_unit_attention

from .gcn_attention import gcn_unit_attention
from .net import Unit2D, conv_init, import_class
from .unit_agcn import unit_gcn

default_backbone_all_layers = [(3, 64, 1), (64, 64, 1), (64, 64, 1), (64, 64, 1), (64, 128,
                                                                                   2), (128, 128, 1),
                               (128, 128, 1), (128, 256, 2), (256, 256, 1), (256, 256, 1)]

default_backbone = [(64, 64, 1), (64, 64, 1), (64, 64, 1), (64, 128,
                                                            2), (128, 128, 1),
                    (128, 128, 1), (128, 256, 2), (256, 256, 1), (256, 256, 1)]


class Model(nn.Module):
    """ Spatial temporal graph convolutional networks
                        for skeleton-based action recognition.

    Input shape:
        Input shape should be (N, C, T, V, M)
        where N is the number of samples,
              C is the number of input channels,
              T is the length of the sequence,
              V is the number of joints or graph nodes,
          and M is the number of people.
    
    Arguments:
        About shape:
            channel (int): Number of channels in the input data
            num_class (int): Number of classes for classification
            window_size (int): Length of input sequence
            num_point (int): Number of joints or graph nodes
            num_person (int): Number of people
        About net:
            use_data_bn: If true, the data will first input to a batch normalization layer
            backbone_config: The structure of backbone networks
        About graph convolution:
            graph: The graph of skeleton, represtented by a adjacency matrix
            graph_args: The arguments of graph
            mask_learning: If true, use mask matrixes to reweight the adjacency matrixes
            use_local_bn: If true, each node in the graph have specific parameters of batch normalzation layer
        About temporal convolution:
            multiscale: If true, use multi-scale temporal convolution
            temporal_kernel_size: The kernel size of temporal convolution
            dropout: The drop out rate of the dropout layer in front of each temporal convolution layer

    """

    def __init__(self,
                 channel,
                 num_class,
                 window_size,
                 num_point,
                 attention,
                 only_attention,
                 tcn_attention,
                 only_temporal_attention,
                 attention_3,
                 relative,
                 kernel_temporal,
                 double_channel,
                 drop_connect,
                 concat_original,
                 dv,
                 dk,
                 Nh,
                 dim_block1,
                 dim_block2,
                 dim_block3,
                 all_layers,
                 data_normalization,
                 visualization,
                 skip_conn,
                 adjacency,
                 bn_flag,
                 weight_matrix,
                 device,
                 n,
                 more_channels,
                 num_person=1,
                 use_data_bn=False,
                 backbone_config=None,
                 graph=None,
                 graph_args=dict(),
                 mask_learning=False,
                 use_local_bn=False,
                 multiscale=False,
                 temporal_kernel_size=9,
                 dropout=0.5):
        super(Model, self).__init__()
        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
            # self.A = torch.from_numpy(self.graph.A).float().cuda(0)
            # self.A = torch.from_numpy(self.graph.A).float()
            self.A = torch.from_numpy(self.graph.A.astype(np.float32))

        self.num_class = num_class
        self.use_data_bn = use_data_bn
        self.multiscale = multiscale
        self.attention = attention
        self.tcn_attention = tcn_attention
        self.drop_connect = drop_connect
        self.more_channels = more_channels
        self.concat_original = concat_original
        self.all_layers = all_layers
        self.dv = dv
        self.num = n
        self.Nh = Nh
        self.dk = dk
        self.data_normalization = data_normalization
        self.skip_conn = skip_conn
        self.visualization = visualization
        print(self.visualization)
        self.double_channel = double_channel
        self.adjacency = adjacency

        # Different bodies share batchNorm parameters or not
        self.M_dim_bn = True

        if self.M_dim_bn:
            self.data_bn = nn.BatchNorm1d(channel * num_point * num_person)
        else:
            self.data_bn = nn.BatchNorm1d(channel * num_point)

        if self.all_layers:
            if not self.double_channel:
                self.starting_ch = 64
            else:
                self.starting_ch = 128
        else:
            if not self.double_channel:
                self.starting_ch = 128
            else:
                self.starting_ch = 256

        kwargs = dict(
            A=self.A,
            mask_learning=mask_learning,
            use_local_bn=use_local_bn,
            dropout=dropout,
            kernel_size=temporal_kernel_size,
            attention=attention,
            only_attention=only_attention,
            tcn_attention=tcn_attention,
            only_temporal_attention=only_temporal_attention,
            attention_3=attention_3,
            relative=relative,
            weight_matrix=weight_matrix,
            device=device,
            more_channels=self.more_channels,
            drop_connect=self.drop_connect,
            data_normalization=self.data_normalization,
            skip_conn=self.skip_conn,
            adjacency=self.adjacency,
            starting_ch=self.starting_ch,
            visualization=self.visualization,
            all_layers=self.all_layers,
            dv=self.dv,
            dk=self.dk,
            Nh=self.Nh,
            num=n,
            dim_block1=dim_block1,
            dim_block2=dim_block2,
            dim_block3=dim_block3,
            num_point=num_point
        )

        if self.multiscale:
            unit = TCN_GCN_unit_multiscale
        else:
            unit = TCN_GCN_unit

        # backbone
        if backbone_config is None:
            if self.all_layers:
                backbone_config = default_backbone_all_layers
            else:
                backbone_config = default_backbone
        self.backbone = nn.ModuleList([
            unit(in_c, out_c, stride=stride, **kwargs)
            for in_c, out_c, stride in backbone_config
        ])
        if self.double_channel:
            backbone_in_c = backbone_config[0][0] * 2
            backbone_out_c = backbone_config[-1][1] * 2
        else:
            backbone_in_c = backbone_config[0][0]
            backbone_out_c = backbone_config[-1][1]
        backbone_out_t = window_size
        backbone = []
        for i, (in_c, out_c, stride) in enumerate(backbone_config):
            if self.double_channel:
                in_c = in_c * 2
                out_c = out_c * 2
            if i == 3 and concat_original:
                backbone.append(unit(in_c + channel, out_c, stride=stride, last=i == len(default_backbone) - 1,
                                     last_graph=(i == len(default_backbone) - 1), layer=i, **kwargs))
            else:
                backbone.append(unit(in_c, out_c, stride=stride, last=i == len(default_backbone) - 1,
                                     last_graph=(i == len(default_backbone) - 1), layer=i, **kwargs))
            if backbone_out_t % stride == 0:
                backbone_out_t = backbone_out_t // stride
            else:
                backbone_out_t = backbone_out_t // stride + 1
        self.backbone = nn.ModuleList(backbone)
        print("self.backbone: ", self.backbone)
        for i in range(0, len(backbone)):
            pytorch_total_params = sum(p.numel() for p in self.backbone[i].parameters() if p.requires_grad)
            print(pytorch_total_params)

        # head

        if not all_layers:
            self.gcn0 = unit_gcn(
                channel,
                backbone_in_c,
                self.A,
                mask_learning=mask_learning,
                use_local_bn=use_local_bn)

            self.tcn0 = Unit2D(backbone_in_c, backbone_in_c, kernel_size=9)

        # tail
        self.person_bn = nn.BatchNorm1d(backbone_out_c)
        self.gap_size = backbone_out_t
        self.fcn = nn.Conv1d(backbone_out_c, num_class, kernel_size=1)
        conv_init(self.fcn)

    def forward(self, x, label, name):
        N, C, T, V, M = x.size()
        print(x.shape)
        if (self.concat_original):
            x_coord = x
            x_coord = x_coord.permute(0, 4, 1, 2, 3).reshape(N * M, C, T, V)

        # data bn
        if self.use_data_bn:
            if self.M_dim_bn:
                x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
            else:
                x = x.permute(0, 4, 3, 1, 2).contiguous().view(N * M, V * C, T)
            print(x.shape)
            x = self.data_bn(x)
            # to (N*M, C, T, V)
            x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(
                N * M, C, T, V)
        else:
            # from (N, C, T, V, M) to (N*M, C, T, V)
            x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)

        # model
        if not self.all_layers:
            x = self.gcn0(x, label, name)
            x = self.tcn0(x)

        for i, m in enumerate(self.backbone):
            if i == 3 and self.concat_original:
                x = m(torch.cat((x, x_coord), dim=1), label, name)
            else:
                x = m(x, label, name)

        # V pooling
        x = F.avg_pool2d(x, kernel_size=(1, V))

        # M pooling
        c = x.size(1)
        t = x.size(2)
        x = x.view(N, M, c, t).mean(dim=1).view(N, c, t)

        # T pooling
        x = F.avg_pool1d(x, kernel_size=x.size()[2])

        # C fcn
        x = self.fcn(x)
        x = F.avg_pool1d(x, x.size()[2:])
        x = x.view(N, self.num_class)
        return x


class TCN_GCN_unit(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 A,
                 attention,
                 only_attention,
                 tcn_attention,
                 only_temporal_attention,
                 relative,
                 device,
                 attention_3,
                 dv,
                 dk,
                 Nh,
                 num,
                 dim_block1,
                 dim_block2,
                 dim_block3,
                 num_point,
                 weight_matrix,
                 more_channels,
                 drop_connect,
                 starting_ch,
                 all_layers,
                 adjacency,
                 data_normalization,
                 visualization,
                 skip_conn,
                 layer=0,
                 kernel_size=9,
                 stride=1,
                 dropout=0.5,
                 use_local_bn=False,
                 mask_learning=False,
                 last=False,
                 last_graph=False

                 ):
        super(TCN_GCN_unit, self).__init__()
        half_out_channel = out_channel / 2
        self.A = A
        self.V = A.size()[-1]
        self.C = in_channel
        self.last = last
        self.data_normalization = data_normalization
        self.skip_conn = skip_conn
        self.num_point = num_point
        self.adjacency = adjacency
        self.last_graph = last_graph
        self.layer = layer
        self.stride = stride
        self.drop_connect = drop_connect
        self.visualization = visualization
        self.device = device
        self.all_layers = all_layers
        self.more_channels = more_channels

        if (out_channel >= starting_ch and attention or (self.all_layers and attention)):

            self.gcn1 = gcn_unit_attention(in_channel, out_channel, dv_factor=dv, dk_factor=dk, Nh=Nh,
                                           complete=True,
                                           relative=relative, only_attention=only_attention, layer=layer, incidence=A,
                                           bn_flag=True, last_graph=self.last_graph, more_channels=self.more_channels,
                                           drop_connect=self.drop_connect, adjacency=self.adjacency, num=num,
                                           data_normalization=self.data_normalization, skip_conn=self.skip_conn,
                                           visualization=self.visualization, num_point=self.num_point)
        else:

            self.gcn1 = unit_gcn(
                in_channel,
                out_channel,
                A,
                use_local_bn=use_local_bn,
                mask_learning=mask_learning)

        if (out_channel >= starting_ch and tcn_attention or (self.all_layers and tcn_attention)):

            if out_channel <= starting_ch and self.all_layers:
                self.tcn1 = tcn_unit_attention_block(out_channel, out_channel, dv_factor=dv,
                                                     dk_factor=dk, Nh=Nh,
                                                     relative=relative, only_temporal_attention=only_temporal_attention,
                                                     dropout=dropout,
                                                     kernel_size_temporal=9, stride=stride,
                                                     weight_matrix=weight_matrix, bn_flag=True, last=self.last,
                                                     layer=layer,
                                                     device=self.device, more_channels=self.more_channels,
                                                     drop_connect=self.drop_connect, n=num,
                                                     data_normalization=self.data_normalization,
                                                     skip_conn=self.skip_conn,
                                                     visualization=self.visualization, dim_block1=dim_block1,
                                                     dim_block2=dim_block2, dim_block3=dim_block3, num_point=self.num_point)
            else:
                self.tcn1 = tcn_unit_attention(out_channel, out_channel, dv_factor=dv,
                                               dk_factor=dk, Nh=Nh,
                                               relative=relative, only_temporal_attention=only_temporal_attention,
                                               dropout=dropout,
                                               kernel_size_temporal=9, stride=stride,
                                               weight_matrix=weight_matrix, bn_flag=True, last=self.last,
                                               layer=layer,
                                               device=self.device, more_channels=self.more_channels,
                                               drop_connect=self.drop_connect, n=num,
                                               data_normalization=self.data_normalization, skip_conn=self.skip_conn,
                                               visualization=self.visualization, num_point=self.num_point)



        else:
            self.tcn1 = Unit2D(
                out_channel,
                out_channel,
                kernel_size=kernel_size,
                dropout=dropout,
                stride=stride)
        if ((in_channel != out_channel) or (stride != 1)):
            self.down1 = Unit2D(
                in_channel, out_channel, kernel_size=1, stride=stride)
        else:
            self.down1 = None

    def forward(self, x, label, name):
        # N, C, T, V = x.size()
        x = self.tcn1(self.gcn1(x, label, name)) + (x if
                                                    (self.down1 is None) else self.down1(x))

        return x


class TCN_GCN_unit_multiscale(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 kernel_size=9,
                 stride=1,
                 **kwargs):
        super(TCN_GCN_unit_multiscale, self).__init__()
        self.unit_1 = TCN_GCN_unit(
            in_channels,
            out_channels / 2,
            A,
            kernel_size=kernel_size,
            stride=stride,
            **kwargs)
        self.unit_2 = TCN_GCN_unit(
            in_channels,
            out_channels - out_channels / 2,
            A,
            kernel_size=kernel_size * 2 - 1,
            stride=stride,
            **kwargs)

    def forward(self, x):
        return torch.cat((self.unit_1(x), self.unit_2(x)), dim=1)
