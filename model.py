import torch
import torch.nn.functional as F
from torch.nn import Parameter
import math
import torch.nn as nn
import numpy as np
import dgl
from dgl.nn.pytorch import edge_softmax, GATConv


def uniform(tensor):
    if tensor is not None:
        nn.init.kaiming_uniform_(tensor)


class GANLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, n_head, attn_drop=0.4,
                 negative_slope=0.2, residual=False, activation=F.elu):
        super(GANLayer, self).__init__()

        self.gat_layers = nn.ModuleList()
        # just only one layer in our paper.
        self.gat_layers.append(GATConv(
            in_channels, out_channels, n_head,
            attn_drop, attn_drop, negative_slope, residual, activation))

    def forward(self, lncrna_x, disease_x, adj):
        index_tuple = np.argwhere(adj == 1)
        lnc_size = lncrna_x.shape[0]
        dis_size = disease_x.shape[0]
        z = torch.cat((lncrna_x, disease_x))
        g = dgl.DGLGraph()
        g.add_nodes(lnc_size + dis_size)
        edge_list = index_tuple
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)

        z = self.gat_layers[0](g, z).flatten(1)
        return z


class MLPLayer(torch.nn.Module):

    def __init__(self, layers):
        super(MLPLayer, self).__init__()
        self.layers = layers
        self.num_layers = len(layers)
        self.mlp_layers = nn.ModuleList()
        for num_layer in range(0, self.num_layers-1): # [64, 16, 1]
            self.mlp_layers.append(torch.nn.Linear(self.layers[num_layer], self.layers[num_layer+1]))


    def forward(self, lncx, disx):
        lnc_size = lncx.shape[0]
        dis_size = disx.shape[0]
        lnc_temp = lncx.repeat(1, dis_size).view(lnc_size * dis_size, -1)
        dis_temp = disx.repeat(lnc_size, 1)
        z = torch.cat([lnc_temp, dis_temp], dim=1).view(lnc_size * dis_size, -1)

        if self.num_layers < 3:
            z = self.mlp_layers[0](z)
        else:
            for num_layer in range(0, self.num_layers - 2):
                z = self.mlp_layers[num_layer](z)
                F.elu(z)
            z = self.mlp_layers[self.num_layers - 2](z)

        output = z.view(lnc_size, dis_size)
        return F.sigmoid(output)


class GANLDAModel(torch.nn.Module):
    def __init__(self, gan_in_channels, gan_out_channels, n_head, attn_drop, mlp_layers):
        super(GANLDAModel, self).__init__()
        self.ganlayer = GANLayer(gan_in_channels, gan_out_channels, n_head, attn_drop)
        self.mlplayer = MLPLayer(mlp_layers)

    def forward(self, lncrna_x, disease_x, adj):
        z = self.ganlayer(lncrna_x, disease_x, adj)
        row_n = lncrna_x.shape[0]
        col_n = disease_x.shape[0]
        feature = torch.split(z, [row_n, col_n], dim=0)
        lnc = feature[0]
        dis = feature[1]

        out = self.mlplayer(lnc, dis)
        return out
