import torch
import torch.nn as nn
import torch.nn.functional as F_func
from .net import Unit2D
from visualization_temporal import visualize
import math
import numpy as np
import time

'''Class that implements the windowed version of temporal transformer.
Function adapted from: https://github.com/leaderj1001/Attention-Augmented-Conv2d
'''
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
multi_matmul = False
dropout = False
scale_norm = False


class tcn_unit_attention_block(nn.Module):
    def __init__(self, in_channels, out_channels, dv_factor, dk_factor, Nh,
                 relative, only_temporal_attention, dropout, kernel_size_temporal, stride, weight_matrix,
                 last, layer, device, more_channels, drop_connect, n, dim_block1, dim_block2, dim_block3,
                 bn_flag=True,
                 shape=25, visualization=False, data_normalization=True, skip_conn=True, more_relative=False):
        super(tcn_unit_attention_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer = layer
        self.drop_connect = drop_connect
        self.visualization = visualization
        self.more_channels = more_channels
        self.data_normalization = data_normalization
        self.skip_conn = skip_conn
        self.only_temporal_att = only_temporal_attention
        self.kernel_size_temporal = kernel_size_temporal
        self.more_relative = more_relative
        self.kernel_size_attention = 9
        self.num = n
        self.dk = int(dk_factor * out_channels)
        if (not self.only_temporal_att):
            self.dv = int(dv_factor * out_channels)
        else:
            self.dv = out_channels
        self.Nh = Nh

        self.bn_flag = bn_flag
        self.shape = shape
        self.relative = relative
        self.stride = stride
        if data_normalization:
            self.data_bn = nn.BatchNorm1d(self.in_channels * 25)

        self.padding = (self.kernel_size_temporal - 1) // 2
        self.bn = nn.BatchNorm2d(out_channels)
        self.weight_matrix = weight_matrix

        if ((self.in_channels != self.out_channels) or (stride != 1)):
            self.down = Unit2D(
                self.in_channels, self.out_channels, kernel_size=1, stride=stride)
        else:
            self.down = None
        self.relu = nn.ReLU(inplace=True)
        self.last = last
        if dropout:
            self.dropout = nn.Dropout(0.25)

        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert self.dk % self.Nh == 0, "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert self.dv % self.Nh == 0, "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"

        # Temporal convolution
        if (not self.only_temporal_att):
            self.tcn_conv = Unit2D(in_channels, out_channels - self.dv, dropout=dropout,
                                   kernel_size=kernel_size_temporal,
                                   stride=self.stride)
        if (self.more_channels):

            self.qkv_conv = nn.Conv2d(self.in_channels, (2 * self.dk + self.dv) * self.Nh // self.num,
                                      kernel_size=(1, stride),
                                      stride=(1, stride),
                                      padding=0)
        else:
            self.qkv_conv = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size=(1, stride),
                                      stride=(1, stride),
                                      padding=0)
        if (self.more_channels):

            self.attn_out = nn.Conv2d(self.dv * self.Nh // self.num, self.dv, kernel_size=1, stride=1)
        else:
            self.attn_out = nn.Conv2d(self.dv, self.dv, kernel_size=1, stride=1)

        if self.out_channels == 64:
            self.block_dim = dim_block1

        if self.out_channels == 128:
            self.block_dim = dim_block2

        if self.out_channels == 256:
            self.block_dim = dim_block3

        if self.relative:
            if self.more_channels:
                self.key_rel = nn.Parameter(

                    torch.randn((2 * self.block_dim - 1, self.dk // self.num), requires_grad=True))

            else:
                self.key_rel = nn.Parameter(

                    torch.randn((2 * self.block_dim - 1, self.dk // Nh), requires_grad=True))


    def forward(self, x):
        # Input x
        # (batch_size, channels, time, joints)
        N1, C, T1, V = x.size()

        x_sum = x

        if (self.data_normalization):
            x = x.permute(0, 1, 3, 2).reshape(N1, C * V, T1)
            x = self.data_bn(x)
            x = x.reshape(N1, C, V, T1).permute(0, 1, 3, 2)

        x = x.permute(0, 3, 1, 2).reshape(-1, C, 1, T1)

        if scale_norm:
            self.scale = ScaleNorm(scale=C ** 0.5)
            x = self.scale(x)

            # Temporal Transformer mechanism is applied separately on each block. Then, the results are concatenated.

        for i in range(0, T1 // self.block_dim):

            block = x[:, :, :, i * self.block_dim: (i * self.block_dim + self.block_dim)]
            N, C, _, T = block.shape
            flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(block, self.dk, self.dv, self.Nh)
            B, self.Nh, C, T = flat_q.size()

            # Calculate the scores, obtained by doing q*k
            # (batch_size, Nh, time, dkh)*(batch_size, Nh, dkh, time) =  (batch_size, Nh, time, time)
            logits = torch.matmul(flat_q.transpose(2, 3), flat_k)

            if self.relative:
                rel_logits = self.relative_logits(q, self.block_dim, i)
                logits_sum = torch.add(logits, rel_logits)

            # Calculate weights
            if self.relative:

                weights = F_func.softmax(logits_sum, dim=-1)

            else:
                weights = F_func.softmax(logits, dim=-1)

            if (self.drop_connect and self.training):
                mask = torch.bernoulli((0.5) * torch.ones(B * self.Nh * T, device=device))
                mask = mask.reshape(B, self.Nh, T).unsqueeze(2).expand(B, self.Nh, T, T)
                weights = weights * mask
                weights = weights / (weights.sum(3, keepdim=True) + 1e-8)

            # attn_out
            # (batch, Nh, time, dvh)
            # weights*V
            # (batch, Nh, time, time)*(batch, Nh, time, dvh)=(batch, Nh, time, dvh)
            attn_out = torch.matmul(weights, flat_v.transpose(2, 3))

            if not self.more_channels:
                attn_out = torch.reshape(attn_out, (B, self.Nh, 1, T, self.dv // self.Nh))
            else:
                attn_out = torch.reshape(attn_out, (B, self.Nh, 1, T, self.dv // self.num))

            # All the blocks are concatenated
            if i == 0:
                attn_out_final = attn_out
            else:
                attn_out_final = torch.cat((attn_out_final, attn_out), dim=3)

        attn_out = attn_out_final.permute(0, 1, 4, 2, 3)

        # combine_heads_2d, combine heads only after having calculated each Z separately
        # (batch, Nh*dv, time, 1)
        attn_out = self.combine_heads_2d(attn_out)
        # Multiply for W0 (batch, out_channels, time, 1) with out_channels=dv
        attn_out = self.attn_out(attn_out)
        attn_out = attn_out.reshape(N1, V, -1, T1 // self.stride).permute(0, 2, 3, 1)

        if self.skip_conn:
            if dropout:
                attn_out = self.dropout(attn_out)

                if (not self.only_temporal_att):
                    x = self.tcn_conv(x_sum)
                    result = torch.cat((x, attn_out), dim=1)
                else:
                    result = attn_out

                result += (x_sum if (self.down is None) else self.down(x_sum))


            else:
                if (not self.only_temporal_att):
                    x = self.tcn_conv(x_sum)
                    result = torch.cat((x, attn_out), dim=1)
                else:
                    result = attn_out

                result += (x_sum if (self.down is None) else self.down(x_sum))


        else:
            result = attn_out

        if (self.bn_flag):
            result = self.bn(result)
        result = self.relu(result)
        return result

    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.qkv_conv(x)
        N, C, H, W = qkv.size()
        if self.more_channels:
            q, k, v = torch.split(qkv, [dk * self.Nh // self.num, dk * self.Nh // self.num, dv * self.Nh // self.num],
                                  dim=1)
        else:
            q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)

        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        q *= dkh ** -0.5
        if self.more_channels:
            flat_q = torch.reshape(q, (N, Nh, dk // self.num, H * W))
            flat_k = torch.reshape(k, (N, Nh, dk // self.num, H * W))
            flat_v = torch.reshape(v, (N, Nh, dv // self.num, H * W))
        else:
            flat_q = torch.reshape(q, (N, Nh, dkh, H * W))
            flat_k = torch.reshape(k, (N, Nh, dkh, H * W))
            flat_v = torch.reshape(v, (N, Nh, dv // self.Nh, H * W))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        B, channels, F, V = x.size()
        ret_shape = (B, Nh, channels // Nh, F, V)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, F, V = x.size()
        ret_shape = (batch, Nh * dv, F, V)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q, blocks_dim, i):
        B, Nh, dk, _, T = q.size()
        # B, Nh, V, T, dk -> B, Nh, F, 1, dk
        q = q.permute(0, 1, 3, 4, 2)
        q = q.reshape(B, Nh, T, dk)
        rel_logits = self.relative_logits_1d(q, self.key_rel)
        # rel_logits_h = self.relative_logits_1d(torch.transpose(q, 2, 3), self.key_rel_h, V, T, Nh, "h")
        return rel_logits

    def relative_logits_1d(self, q, rel_k):
        # compute relative logits along one dimension
        # (B, Nh,  1, V, channels // Nh)*(2 * K - 1, self.dk // Nh)
        # (B, Nh,  1, V, 2 * K - 1)
        # print("case", case)
        # print("input relative logits_q ", q.shape)
        # print("input relative logits_rel ", rel_k.shape)

        rel_logits = torch.einsum('bhld,md->bhlm', q, rel_k)

        rel_logits = self.rel_to_abs(rel_logits)
        B, Nh, L, L = rel_logits.size()

        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()
        col_pad = torch.zeros((B, Nh, L, 1)).to(x)
        x = torch.cat((x, col_pad), dim=3)
        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).to(x)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)

        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x


class ScaleNorm(nn.Module):
    """ScaleNorm"""

    def __init__(self, scale, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.scale = scale

        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=1, keepdim=True).clamp(min=self.eps)
        return x * norm
