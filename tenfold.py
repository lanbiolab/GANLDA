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
all_tpr = []
all_fpr = []
all_recall = []
all_precision = []
all_accuracy = []


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

    # evaluation start
    output = out.cpu().data.numpy()
    score_matrix = output

    zero_matrix = np.zeros((score_matrix.shape[0], score_matrix.shape[1])).astype('int64')
    score_matrix_temp = score_matrix.copy()
    score_matrix = score_matrix_temp + zero_matrix
    minvalue = np.min(score_matrix)
    score_matrix[np.where(roc_lncrna_disease_matrix == 2)] = minvalue - 10

    sorted_lncrna_disease_matrix, sorted_score_Matrix = methods.sort_matrix(score_matrix, roc_lncrna_disease_matrix)

    tpr_list = []
    fpr_list = []
    recall_list = []
    precision_list = []
    accuracy_list = []
    for cutoff in range(sorted_lncrna_disease_matrix.shape[0]):
        P_matrix = sorted_lncrna_disease_matrix[0:cutoff + 1, :]
        N_matrix = sorted_lncrna_disease_matrix[cutoff + 1:sorted_lncrna_disease_matrix.shape[0] + 1, :]
        TP = np.sum(P_matrix == 1)
        FP = np.sum(P_matrix == 0)
        TN = np.sum(N_matrix == 0)
        FN = np.sum(N_matrix == 1)
        tpr = TP / (TP + FN)
        fpr = FP / (FP + TN)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        recall_list.append(recall)
        precision_list.append(precision)
        accuracy = (TN + TP) / (TN + TP + FN + FP)

        accuracy_list.append(accuracy)
    all_tpr.append(tpr_list)
    all_fpr.append(fpr_list)
    all_recall.append(recall_list)
    all_precision.append(precision_list)
    all_accuracy.append(accuracy_list)
tpr_arr = np.array(all_tpr)
fpr_arr = np.array(all_fpr)
recall_arr = np.array(all_recall)
precision_arr = np.array(all_precision)
accuracy_arr = np.array(all_accuracy)

mean_cross_tpr = np.mean(tpr_arr, axis=0)
mean_cross_fpr = np.mean(fpr_arr, axis=0)
mean_cross_recall = np.mean(recall_arr, axis=0)
mean_cross_precision = np.mean(precision_arr, axis=0)
mean_cross_accuracy = np.mean(accuracy_arr, axis=0)

roc_auc = np.trapz(mean_cross_tpr, mean_cross_fpr)

plt.plot(mean_cross_fpr, mean_cross_tpr, label='mean ROC=%0.4f' % roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=0)
print("runtime over, now is :")
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
plt.show()
