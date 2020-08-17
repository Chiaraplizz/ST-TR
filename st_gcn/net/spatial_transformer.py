import torch
import torch.nn as nn
import torch.nn.functional as F
from visualization_temporal import visualize
import numpy as np

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
multi_matmul = False

'''
This class implements Spatial Transformer. 
Function adapted from: https://github.com/leaderj1001/Attention-Augmented-Conv2d

'''


class spatial_attention(nn.Module):
    def __init__(self, in_channels, kernel_size, dk, dv, Nh, complete, relative, layer, A, more_channels, drop_connect,
                 adjacency, num,
                 shape=25, stride=1,
                 last_graph=False, data_normalization=True, skip_conn=True, visualization=True):
        super(spatial_attention, self).__init__()
        self.in_channels = in_channels
        self.complete = complete
        self.kernel_size = kernel_size
        self.dk = dk
        self.dv = dv
        self.num = num
        self.layer = layer
        self.more_channels = more_channels
        self.drop_connect = drop_connect
        self.visualization = visualization
        self.data_normalization = data_normalization
        self.skip_conn = skip_conn
        self.adjacency = adjacency
        self.Nh = Nh
        self.A = A[0] + A[1] + A[2]
        if self.adjacency:
            self.mask = nn.Parameter(torch.ones(self.A.size()))
        self.shape = shape
        self.relative = relative
        self.last_graph = last_graph
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2

        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert self.dk % self.Nh == 0, "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert self.dv % self.Nh == 0, "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."


        if (self.more_channels):

            self.qkv_conv = nn.Conv2d(self.in_channels, (2 * self.dk + self.dv) * self.Nh // self.num,
                                      kernel_size=self.kernel_size,
                                      stride=stride,
                                      padding=self.padding)
        else:
            self.qkv_conv = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size=self.kernel_size,
                                      stride=stride,
                                      padding=self.padding)
        if (self.more_channels):

            self.attn_out = nn.Conv2d(self.dv * self.Nh // self.num, self.dv, kernel_size=1, stride=1)
        else:
            self.attn_out = nn.Conv2d(self.dv, self.dv, kernel_size=1, stride=1)

        if self.relative:
            # Two parameters are initialized in order to implement relative positional encoding
            # One weight repeated over the diagonal
            # V^2-V+1 paramters in positions outside the diagonal
            if self.more_channels:
                self.key_rel = nn.Parameter(torch.randn(((25 ** 2) - 25, self.dk // self.num), requires_grad=True))
            else:
                self.key_rel = nn.Parameter(torch.randn(((25 ** 2) - 25, self.dk // Nh), requires_grad=True))
            if self.more_channels:
                self.key_rel_diagonal = nn.Parameter(torch.randn((1, self.dk // self.num), requires_grad=True))
            else:
                self.key_rel_diagonal = nn.Parameter(torch.randn((1, self.dk // self.Nh), requires_grad=True))

    def forward(self, x, label, name):
        # Input x
        # (batch_size, channels, 1, joints)
        B, _, T, V = x.size()

        # flat_q, flat_k, flat_v
        # (batch_size, Nh, dvh or dkh, joints)
        # dvh = dv / Nh, dkh = dk / Nh
        # q, k, v obtained by doing 2D convolution on the input (q=XWq, k=XWk, v=XWv)
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)

        # Calculate the scores, obtained by doing q*k
        # (batch_size, Nh, joints, dkh)*(batch_size, Nh, dkh, joints) =  (batch_size, Nh, joints,joints)
        # The multiplication can also be divided (multi_matmul) in case of space problems
        if (multi_matmul):
            for i in range(0, 5):
                flat_q_5 = flat_q[:, :, :, (5 * i):(5 * (i + 1))]
                product = torch.matmul(flat_q_5.transpose(2, 3), flat_k)
                if (i == 0):
                    logits = product
                else:
                    logits = torch.cat((logits, product), dim=2)
        else:
            logits = torch.matmul(flat_q.transpose(2, 3), flat_k)

        # In this version, the adjacency matrix is weighted and added to the attention logits of transformer to add
        # information of the original skeleton structure
        if (self.adjacency):
            self.A = self.A.cuda(device)
            logits = logits.reshape(-1, V, V)
            M, V, V = logits.shape
            A = self.A
            A *= self.mask
            A = A.unsqueeze(0).expand(M, V, V)
            logits += A
            logits = logits.reshape(B, self.Nh, V, V)

        # Relative positional encoding is used or not
        if self.relative:
            rel_logits = self.relative_logits(q)
            logits_sum = torch.add(logits, rel_logits)

        # Calculate weights
        if self.relative:
            weights = F.softmax(logits_sum, dim=-1)
        else:
            weights = F.softmax(logits, dim=-1)

        # Drop connect implementation to avoid overfitting
        if (self.drop_connect and self.training):
            mask = torch.bernoulli((0.5) * torch.ones(B * self.Nh * V, device=device))
            mask = mask.reshape(B, self.Nh, V).unsqueeze(2).expand(B, self.Nh, V, V)
            weights = weights * mask
            weights = weights / (weights.sum(3, keepdim=True) + 1e-8)

        # Code for the visualization of attention logits
        if (self.visualization):
            weights = weights.reshape(2, -1, self.Nh, V, V)
            logits = logits.reshape(2, -1, self.Nh, V, V)
            if self.relative:
                logits_sum = logits_sum.reshape(2, -1, self.Nh, V, V)
                rel_logits = rel_logits.reshape(2, -1, self.Nh, V, V)
                logits = logits.reshape(2, -1, self.Nh, V, V)
                min_v = min(float(torch.min(logits[:, :, 0, :, :])), float(torch.min(logits_sum[:, :, 0, :, :])),
                            float(torch.min(rel_logits[:, :, 0, :, :])))
                max_v = max(float(torch.max(logits[:, :, 0, :, :])), float(torch.max(logits_sum[:, :, 0, :, :])),
                            float(torch.max(rel_logits[:, :, 0, :, :])))
                # visualize(logits_sum, str(label), 'logits_sum', self.layer,name)
                # visualize((logits_sum - min_v) / (max_v - min_v), str(label), 'logits_sum_norm', self.layer, name)
                # visualize(rel_logits, str(label), 'rel_logits', self.layer,name)
                visualize(weights, (rel_logits - min_v) / (max_v - min_v), str(label), 'rel_logits_norm', self.layer,
                          name)
            weights = weights.reshape(B, self.Nh, V, V)

        # attn_out
        # (batch, Nh, joints, dvh)
        # weights*V
        # (batch, Nh, joints, joints)*(batch, Nh, joints, dvh)=(batch, Nh, joints, dvh)
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))

        if not self.more_channels:
            attn_out = torch.reshape(attn_out, (B, self.Nh, T, V, self.dv // self.Nh))
        else:
            attn_out = torch.reshape(attn_out, (B, self.Nh, T, V, self.dv // self.num))

        attn_out = attn_out.permute(0, 1, 4, 2, 3)

        # combine_heads_2d, combine heads only after having calculated each Z separately
        # (batch, Nh*dv, 1, joints)
        attn_out = self.combine_heads_2d(attn_out)

        # Multiply for W0 (batch, out_channels, 1, joints) with out_channels=dv
        attn_out = self.attn_out(attn_out)
        return attn_out

    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.qkv_conv(x)
        # T=1 in this case, because we are considering each frame separately
        N, _, T, V = qkv.size()

        # if self.more_channels=True, to each head is assigned dk*self.Nh//self.num channels
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
            flat_q = torch.reshape(q, (N, Nh, dk // self.num, T * V))
            flat_k = torch.reshape(k, (N, Nh, dk // self.num, T * V))
            flat_v = torch.reshape(v, (N, Nh, dv // self.num, T * V))
        else:
            flat_q = torch.reshape(q, (N, Nh, dkh, T * V))
            flat_k = torch.reshape(k, (N, Nh, dkh, T * V))
            flat_v = torch.reshape(v, (N, Nh, dv // self.Nh, T * V))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        B, channels, T, V = x.size()
        ret_shape = (B, Nh, channels // Nh, T, V)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, T, V = x.size()
        ret_shape = (batch, Nh * dv, T, V)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        B, Nh, dk, T, V = q.size()
        q = torch.transpose(q, 2, 4).transpose(2, 3)
        q_first = q.unsqueeze(4).expand((B, Nh, T, V, V - 1, dk))
        q_first = torch.reshape(q_first, (B * Nh * T, -1, dk))

        # q used to multiply for the embedding of the parameter on the diagonal
        q = torch.reshape(q, (B * Nh * T, V, dk))
        # key_rel_diagonal: (1, dk) -> (V, dk)
        param_diagonal = self.key_rel_diagonal.expand((V, dk))
        rel_logits = self.relative_logits_1d(q_first, q, self.key_rel, param_diagonal, T, V, Nh)
        return rel_logits

    def relative_logits_1d(self, q_first, q, rel_k, param_diagonal, T, V, Nh):
        # compute relative logits along one dimension
        # (B*Nh*1,V^2-V, self.dk // Nh)*(V^2 - V, self.dk // Nh)

        # (B*Nh*1, V^2-V)
        rel_logits = torch.einsum('bmd,md->bm', q_first, rel_k)
        # (B*Nh*1, V)
        rel_logits_diagonal = torch.einsum('bmd,md->bm', q, param_diagonal)

        # reshapes to obtain Srel
        rel_logits = self.rel_to_abs(rel_logits, rel_logits_diagonal)

        rel_logits = torch.reshape(rel_logits, (-1, Nh, V, V))
        return rel_logits

    def rel_to_abs(self, rel_logits, rel_logits_diagonal):
        B, L = rel_logits.size()
        B, V = rel_logits_diagonal.size()

        # (B, V-1, V) -> (B, V, V)
        rel_logits = torch.reshape(rel_logits, (B, V - 1, V))
        row_pad = torch.zeros(B, 1, V).to(rel_logits)
        rel_logits = torch.cat((rel_logits, row_pad), dim=1)

        # concat the other embedding on the left
        # (B, V, V) -> (B, V, V+1) -> (B, V+1, V)
        rel_logits_diagonal = torch.reshape(rel_logits_diagonal, (B, V, 1))
        rel_logits = torch.cat((rel_logits_diagonal, rel_logits), dim=2)
        rel_logits = torch.reshape(rel_logits, (B, V + 1, V))

        # slice
        flat_sliced = rel_logits[:, :V, :]
        final_x = torch.reshape(flat_sliced, (B, V, V))
        return final_x
