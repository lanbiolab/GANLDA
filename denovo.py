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

parser.add_argument('--gan_in', type=int, default=32,
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

with h5py.File('lncRNA_disease_Associations.h5', 'r') as hf:
    lncrna_disease_matrix = hf['rating'][:]
    lncrna_disease_matrix_val =  lncrna_disease_matrix.copy()

all_tpr = []
all_fpr = []

all_recall = []
all_precision = []
all_accuracy = []

# denovo start
for i in range(412):
    new_lncrna_disease_matrix = lncrna_disease_matrix.copy()
    roc_lncrna_disease_matrix = lncrna_disease_matrix.copy()
    if ((False in (new_lncrna_disease_matrix[:,i]==0))==False):
        continue
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

    for epoch in range(1, 1000):
        out, loss = methods.train(rel_matrix, ganlda_model, optimizer, lncx, disx,
                                  adj)  # label, ganlda_model, optimizer, lncx, disx, adj
        print('the ' + str(epoch) + ' times loss is' + str(loss))

    score_matrix = out.cpu().data.numpy()

    # evaluation start
    sort_index = np.argsort(-score_matrix[:,i],axis=0)
    sorted_lncrna_disease_row = roc_lncrna_disease_matrix[:,i][sort_index]
    tpr_list = []
    fpr_list = []

    recall_list = []
    precision_list = []

    accuracy_list = []
    for cutoff in range(1, 241):
        P_vector = sorted_lncrna_disease_row[0:cutoff]
        N_vector = sorted_lncrna_disease_row[cutoff:]
        TP = np.sum(P_vector == 1)
        FP = np.sum(P_vector == 0)
        TN = np.sum(N_vector == 0)
        FN = np.sum(N_vector == 1)
        tpr = TP/(TP+FN)
        fpr = FP/(FP+TN)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        recall = TP/(TP+FN)
        precision = TP/(TP+FP)

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

mean_denovo_recall = np.mean(recall_arr,axis=0)
mean_denovo_precision = np.mean(precision_arr,axis=0)

mean_denovo_tpr = np.mean(tpr_arr,axis=0)
mean_denovo_fpr = np.mean(fpr_arr,axis=0)
mean_denovo_accuracy = np.mean(accuracy_arr,axis=0)

roc_auc = metrics.auc(mean_denovo_fpr, mean_denovo_tpr)
plt.plot(mean_denovo_fpr,mean_denovo_tpr, label='mean ROC=%0.4f'%roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
