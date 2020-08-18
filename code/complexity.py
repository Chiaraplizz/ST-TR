import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import configargparse as argparse
from arg_types import arg_boolean
from original_code.tools.net.temporal_transformer import tcn_unit_attention
from original_code.tools.net.gcn_attention import gcn_unit_attention
from original_code.tools.net.net import Unit2D, import_class
from original_code.tools.net import unit_gcn

import networkx as nx

device = "cuda:0" if torch.cuda.is_available() else "cpu"
graph = nx.Graph()
Graph = import_class("st_gcn.graph.NTU_RGB_D")
graph = Graph('spatial')
A = torch.from_numpy(graph.A).float()


class Block(nn.Module):
    def __init__(self, model_type, channels, head):
        super(Block, self).__init__()

        if model_type == "S+conv":
            self.gcn = gcn_unit_attention(channels, channels, dv_factor=0.25, dk_factor=0.25, Nh=head,
                                          complete=True,
                                          relative=False, only_attention=True, layer=0, incidence=A, num=n,
                                          bn_flag=True, last_graph=False, more_channels=False,
                                          drop_connect=False, adjacency=1,
                                          data_normalization=True, skip_conn=True,
                                          visualization=False)
            self.tcn = Unit2D(
                channels,
                channels,
                kernel_size=9,
                dropout=0.25,
                stride=1)
        if model_type == "conv+T":
            self.gcn = unit_gcn(
                channels,
                channels,
                A,
                use_local_bn=False,
                mask_learning=True)
            self.tcn = tcn_unit_attention(channels, channels, dv_factor=0.25,
                                          dk_factor=0.25, Nh=8,
                                          relative=False, only_temporal_attention=True,
                                          dropout=False, n=n,
                                          kernel_size_temporal=9, stride=1,
                                          weight_matrix=False, bn_flag=True, last=False, layer=0,
                                          device=device, more_channels=False,
                                          drop_connect=False,
                                          data_normalization=True, skip_conn=True,
                                          visualization=False)
        if model_type == "conv+conv":
            self.gcn = unit_gcn(
                channels,
                channels,
                A,
                use_local_bn=False,
                mask_learning=True)
            self.tcn = Unit2D(
                channels,
                channels,
                kernel_size=9,
                dropout=0.25,
                stride=1)
        if model_type == "agcn+conv":
            self.gcn = unit_gcn(
                    channels,
                    channels,
                    A,
                    use_local_bn=False,
                    mask_learning=True)
            self.tcn = Unit2D(
                    channels,
                    channels,
                    kernel_size=9,
                    dropout=0.25,
                    stride=1)

    def forward(self, x):
        x = self.tcn(self.gcn(x, "",""))
        return x


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default='/multiverse/storage/plizzari/code/eccv/complexity.csv')
    parser.add_argument('--input_height', type=int, default=1)
    parser.add_argument('--model_type', type=str, default='agcn+conv', help="GC or ST or TT or TC")
    parser.add_argument('--channels', type=int, default=None)
    parser.add_argument('--training', type=arg_boolean, default=False)
    parser.add_argument('--head', type=float, default=None)
    parser.add_argument('--n', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=2)

    args, _ = parser.parse_known_args()
    return args


def trial(file, model_type, batch_size, channels, head, n, input_height, training=False, save=True):
    torch.manual_seed(42)
    rand_x = torch.rand((batch_size * 2, channels, 300, 25), device=device)
    # rand_x = torch.randint(0, input_height, size=[batch_size * 2, channels, 300, 25], device=device)
    rand_y = torch.randint(0, input_height, size=[batch_size * 2, channels, 300, 25], device=device)

    print(model_type)

    if model_type=="conv+conv" or model_type=="S+conv" or model_type=="conv+T" or model_type=="agcn+conv":
        net= Block(model_type, channels, head)



    if model_type == "ST":
        net = gcn_unit_attention(channels, channels, dv_factor=0.25, dk_factor=0.25, Nh=head,
                                 complete=True,
                                 relative=False, only_attention=True, layer=0, incidence=A, num=n,
                                 bn_flag=True, last_graph=False, more_channels=False,
                                 drop_connect=False, adjacency=1,
                                 data_normalization=True, skip_conn=True,
                                 visualization=False)

    if model_type == "GC":
        net = unit_gcn(
            channels,
            channels,
            A,
            use_local_bn=False,
            mask_learning=True)

    if model_type == "TT":
        net = tcn_unit_attention(channels, channels, dv_factor=0.25,
                                 dk_factor=0.25, Nh=8,
                                 relative=False, only_temporal_attention=True,
                                 dropout=False, n=n,
                                 kernel_size_temporal=9, stride=1,
                                 weight_matrix=False, bn_flag=True, last=False, layer=0,
                                 device=device, more_channels=False,
                                 drop_connect=False,
                                 data_normalization=True, skip_conn=True,
                                 visualization=False)
    if model_type == "TC":
        net = Unit2D(
            channels,
            channels,
            kernel_size=9,
            dropout=0.25,
            stride=1)

    net = net.to(device)

    if training:
        net.train()
    else:
        net.eval()

    fw_start = time.time()
    if model_type == "GC" or model_type == "ST":
        output = net(rand_x, "", "")
    else:
        output = net(rand_x)

    fw_time = time.time() - fw_start
    sample_time = fw_time / rand_x.shape[0]
    params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    if True:
        with open(file, 'a') as f:
            f.write(f"{model_type};{batch_size};"
                    f"{channels};{head};{n};{sample_time};{params}\n".replace(".", ","))


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    params = get_params()
    os.makedirs(os.path.dirname(params.output_file), exist_ok=True)
    # Write headers
    with open(params.output_file, 'a') as f:
        f.write("model_type;batch_size;channels;head;n;sample_time;params\n")
    # Burn-in
    print("Burn-in")
    for _ in tqdm(range(20)):
        _ = trial(params.output_file, "agcn+conv", params.batch_size, 64, 8, 1, 100, training=False, save=False)

    for model_type in ["agcn+conv", "GC"]:
        print(f"Analyzing {model_type}")
        pbar = tqdm()
        for channels in {64, 128, 256, 512}:
            for head in {1}:
                for n in {8}:
                    if not params.training:
                        with torch.no_grad():
                            trial(params.output_file, model_type, params.batch_size, channels, head, input_height=60,
                                  n=n, training=False, save=True)
                    else:
                        trial(params, model_type, params.batch_size, channels, head, input_height=60,
                              n=n, training=True, save=True)
