from sklearn import metrics
import torch
import matplotlib.pyplot as plt
import numpy as np
import h5py
import methods
from model import GANLDAModel
import itertools
from sklearn.decomposition import PCA
import argparse


parser = argparse.ArgumentParser(description='GANLDA Example')
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

# load feature data
with h5py.File('lncRNA_Features.h5', 'r') as hf:
    lncx = hf['infor'][:]
    pca = PCA(n_components=gan_in_channels)
    lncx = pca.fit_transform(lncx)
    lncx = torch.Tensor(lncx)
with h5py.File('disease_Features.h5', 'r') as hf:
    diseasex = hf['infor'][:]
    pca = PCA(n_components=gan_in_channels)
    diseasex = pca.fit_transform(diseasex)
    disx = torch.Tensor(diseasex)

# load lncRNA-disease associations
with h5py.File('lncRNA_disease_Associations.h5', 'r') as hf:
    lncrna_disease_matrix = hf['rating'][:]
    lncrna_disease_matrix_val =  lncrna_disease_matrix.copy()

# denovo start
for i in range(lncrna_disease_matrix.shape[1]):
    new_lncrna_disease_matrix = lncrna_disease_matrix.copy()
    roc_lncrna_disease_matrix = lncrna_disease_matrix.copy()
    
    # de novo test for a new disease
    new_lncrna_disease_matrix[:,i] = 0

    rel_matrix = new_lncrna_disease_matrix
    row_n = rel_matrix.shape[0]
    col_n = rel_matrix.shape[1]
    temp_l = np.zeros((row_n, row_n))
    temp_d = np.zeros((col_n, col_n))
    adj = np.vstack((np.hstack((temp_l, rel_matrix)), np.hstack((rel_matrix.T, temp_d))))

    ganlda_model = GANLDAModel(gan_in_channels, gan_out_channels, n_head, attn_drop, mlp_layers)
    optimizer = torch.optim.Adam(ganlda_model.parameters(), lr=lr,
                                 weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 800], gamma=0.8)
    for epoch in range(1, 1000):
        # train
        out, loss = methods.train(rel_matrix, ganlda_model, optimizer, lncx, disx,
                                  adj)  # label, ganlda_model, optimizer, lncx, disx, adj
        print('the ' + str(epoch) + ' times loss is' + str(loss))
        scheduler.step()
    
    # score matrix
    score_matrix = out.cpu().data.numpy()

   
