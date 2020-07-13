from __future__ import print_function
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import h5py
import time
import methods
from model import GANLDAModel
import itertools
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser(description='GANLDA ten-fold')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--gan_in', type=int, default=128,
                        help='PCA embedding size.')
parser.add_argument('--gan_out', type=int, default=8,
                        help='GAN embedding size.')
parser.add_argument('--n_head', type=int, default=8,
                        help='GAN head number.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.00005,
                    help='Weight decay.')
parser.add_argument('--att_drop_rate', type=float, default=0.4,
                    help='GAT Dropout rate.')
parser.add_argument('--mlp_layers', nargs='?', type=list, default=[128,64,64,1],
                        help="Size of each mlp layer.")
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

gan_in_channels = args.gan_in
gan_out_channels = args.gan_out
n_head = args.n_head
lr = args.lr
weight_decay = args.weight_decay
attn_drop = args.att_drop_rate
mlp_layers = args.mlp_layers

torch.manual_seed(args.seed)

device = torch.device('cpu')
with h5py.File('lncRNA_disease_Associations.h5', 'r') as hf:
    lncrna_disease_matrix = hf['rating'][:]
    lncrna_disease_matrix_val = lncrna_disease_matrix.copy()
index_tuple = (np.where(lncrna_disease_matrix == 1))
one_list = list(zip(index_tuple[0], index_tuple[1]))
random.shuffle(one_list)
split = math.ceil(len(one_list) / 10)

# load feature data
with h5py.File('lncRNA_Features.h5', 'r') as hf:
    lncx = hf['infor'][:]
    pca = PCA(n_components=gan_in_channels)
    lncx = pca.fit_transform(lncx)
    lncx = torch.Tensor(lncx)
with h5py.File('disease_Features.h5', 'r') as hf:
    disx = hf['infor'][:]
    pca = PCA(n_components=gan_in_channels)
    disx = pca.fit_transform(disx)
    disx = torch.Tensor(disx)

# 10-fold start
for i in range(0, len(one_list), split):
    
    # ganlda model
    ganlda_model = GANLDAModel(gan_in_channels, gan_out_channels, n_head, attn_drop, mlp_layers)
    optimizer = torch.optim.Adam(ganlda_model.parameters(), lr=lr,
                                 weight_decay=weight_decay)

    train_index = one_list[i:i + split]
    new_lncrna_disease_matrix = lncrna_disease_matrix.copy()

    for index in train_index:
        new_lncrna_disease_matrix[index[0], index[1]] = 0  # train set
    roc_lncrna_disease_matrix = new_lncrna_disease_matrix + lncrna_disease_matrix

    rel_matrix = new_lncrna_disease_matrix
    row_n = rel_matrix.shape[0]
    col_n = rel_matrix.shape[1]
    temp_l = np.zeros((row_n, row_n))
    temp_d = np.zeros((col_n, col_n))
    adj = np.vstack((np.hstack((temp_l, rel_matrix)), np.hstack((rel_matrix.T, temp_d))))
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 800], gamma=0.8)

    for epoch in range(1, 1000):
        out, loss = methods.train(rel_matrix, ganlda_model,optimizer, lncx, disx, adj) # label, ganlda_model, optimizer, lncx, disx, adj
        print('the ' + str(epoch) + ' times loss is ' + str(loss))
        scheduler.step()

    output = out.cpu().data.numpy()
    
    # the score matrix
    score_matrix = output
    
    

  
