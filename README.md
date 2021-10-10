# GANLDA
The implementation of “GANLDA: Graph attention network for lncRNA-disease associations prediction”, Wei Lan, Ximin Wu, Qingfeng Chen, Wei Peng, Jianxin Wang, Yiping Phoebe Chen. Neurocomputing, 2021. The GAT layer is based on [DGL](https://github.com/dmlc/dgl).  

## Requirement

- Python 3.6

- Numpy

- dgl

- Sklearn

- scipy

- matplotlib

- random

- math

- h5py

- pickle

- torch

- argparse

- itertools

## Data 
### The diseases and lncRNAs association matrix: lncRNA_disease_Associations.h5
### The diseases features: disease_Features.h5
### The lncRNAs features: lncRNA_Features.h5
### The lncRNAs name: lncRNA-name.xlsx
### The disease doid: doid.xlsx

## Run
### The ganlda init program entry: ganlda_init.py
### The 10-fold program entry: tenfold.py
### The denovo program entry: denovo.py

## Obtain the score matrix
### If you want to obtain score matrix by GANLDA framework, please run ganlda_init.py directly.



