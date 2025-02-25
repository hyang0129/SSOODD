
from PIL import Image
import torch
import torchvision
import numpy as np
import torch.nn.functional as F
from torchvision import transforms as T

import sys
sys.path.append('.')

from goo.networks.paws_encoders import ProjectionHead

import os

from pdb import set_trace as pb
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import normalize
from sklearn import metrics
import faiss

import pandas as pd
from sklearn.preprocessing import normalize

def eval_ood_KNN(indist_labelled, indist_val, outdist, k=1):

    indist_labelled = normalize(indist_labelled, axis=1)
    indist_val = normalize(indist_val, axis=1)
    outdist = normalize(outdist, axis=1)

    index = faiss.IndexFlatIP(indist_labelled.shape[1])
    index.add(np.ascontiguousarray(indist_labelled))
    D_indist, I = index.search(indist_val, k)

    D_indist = D_indist.min(axis=1)

    index = faiss.IndexFlatIP(indist_labelled.shape[1])
    index.add(np.ascontiguousarray(indist_labelled))
    D_outdist, I = index.search(outdist, k)

    D_outdist = D_outdist.min(axis=1)

    labels = [np.zeros(D_outdist.shape[0]), np.ones(D_indist.shape[0])]
    labels = np.concatenate(labels)
    dists = np.concatenate([D_outdist, D_indist]).squeeze()

    aucroc = metrics.roc_auc_score(labels, dists)

    fp95 = np.quantile(D_indist, 0.05)
    fp95_in = np.mean(D_outdist > fp95)
    return aucroc, fp95_in

# ============================================

OOD_data_orig = np.load('weights/cifar-100-epoch=0-step=0-_predict_head_semi_0.npy', allow_pickle=True)
OOD_data_orig = normalize(OOD_data_orig, axis=1).astype(np.float32)
ID_val_data = np.load('weights/cifar-10-epoch=0-step=0-_test_head_semi_0.npy', allow_pickle=True)
ID_val_data = normalize(ID_val_data, axis=1)
ID_training_data_orig = ID_val_data[:40].astype(np.float32)
ID_val_data_orig = ID_val_data[40:].astype(np.float32)

print(eval_ood_KNN(ID_training_data_orig, ID_val_data_orig, OOD_data_orig, k=1))
# ============================================

# initialise stuff
exemplar_sampling = pd.read_csv('weights/SKMPS_CIFAR-10.csv')
head_path = 'weights/molar-CIFAR-10-head.ckpt'
# head_path = 'weights/molar-SS-CIFAR-10-head.ckpt'
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

head_params = torch.load(head_path)

head = ProjectionHead(hidden_dim=head_params['fc.fc1.weight'].shape[1], output_dim=head_params['fc.fc3.weight'].shape[0])
head.load_state_dict(torch.load(head_path, weights_only=True))

orig_mod = torch.nn.Sequential(dinov2_vits14, head)

orig_mod.cuda()
orig_mod.eval()

# pb()
# test2 = torch.zeros(1, 384)
# head(test2.cuda())

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=256, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
    torchvision.transforms.CenterCrop(size=224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

knn_alpha = 1.0
knn_k = 1

ID_training = torchvision.datasets.CIFAR10(download=True, train=True, root='datasets/', transform=transform)
# np.random.seed(0)
# ID_training.data = ID_training.data[np.random.choice(ID_training.data.shape[0], size=int(knn_alpha*ID_training.data.shape[0]), replace=False)]
ID_training.data = ID_training.data[exemplar_sampling['indices']]

ID_val = torchvision.datasets.CIFAR10(download=True, train=False, root='datasets/', transform=transform)

OOD_train = torchvision.datasets.CIFAR100(download=True, train=True, root='datasets/', transform=transform)
OOD_val = torchvision.datasets.CIFAR100(download=True, train=False, root='datasets/', transform=transform)
OOD = torch.utils.data.ConcatDataset([OOD_train, OOD_val])

ID_training = DataLoader(ID_training, batch_size=64, shuffle=False, num_workers=4, drop_last=False)
ID_val = DataLoader(ID_val, batch_size=64, shuffle=False, num_workers=4, drop_last=False)
OOD = DataLoader(OOD, batch_size=64, shuffle=False, num_workers=4, drop_last=False)

ID_training_data = []
ID_val_data = []
OOD_data = []

torch.set_float32_matmul_precision('medium')

for inputs, targets in tqdm(ID_training, desc="Training", leave=False):
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            inputs = inputs.cuda()
            res = dinov2_vits14(inputs)
            with torch.amp.autocast(enabled=False, device_type="cuda"):
                res = head(res)
            ID_training_data.append(res.cpu())

for inputs, targets in tqdm(ID_val, desc="Training", leave=False):
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            inputs = inputs.cuda()
            ID_val_data.append(orig_mod(inputs).cpu())

for inputs, targets in tqdm(OOD, desc="Training", leave=False):
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            inputs = inputs.cuda()
            OOD_data.append(orig_mod(inputs).cpu())

# ====================================================

OOD_data_orig_unnorm = np.load('weights/cifar-100-epoch=0-step=0-_predict_head_semi_0.npy', allow_pickle=True)
ID_val_data_unnorm = np.load('weights/cifar-10-epoch=0-step=0-_test_head_semi_0.npy', allow_pickle=True)
ID_val_data_unnorm = ID_val_data_unnorm[40:]

ID_val_data_mine = torch.cat(ID_val_data, dim=0)
OOD_data_mine = torch.cat(OOD_data, dim=0)

# OOD_data_orig_unnorm[39875] - OOD_data_mine[0].numpy()
# OOD_data_orig_unnorm[508] - OOD_data_mine[1].numpy()

# ID_val_data_unnorm[5694] - ID_val_data_mine[0].numpy()

# ====================================================

ID_training_data = torch.cat(ID_training_data, dim=0)
ID_training_data = torch.nn.functional.normalize(ID_training_data, dim=1).float().numpy()
# ID_training_data[33] - ID_training_data_orig[0]

ID_val_data = torch.cat(ID_val_data, dim=0).float()
ID_val_data = torch.nn.functional.normalize(ID_val_data, dim=1).float().numpy()
# ID_val_data_orig[5694] - ID_val_data[0]

OOD_data = torch.cat(OOD_data, dim=0).float()
OOD_data = torch.nn.functional.normalize(OOD_data, dim=1).float().numpy()

# np.absolute(OOD_data[1,None,:] - OOD_data_orig).sum(axis=1).argmin()
# OOD_data_orig[39875] - OOD_data[0]
# OOD_data_orig[508] - OOD_data[1]

print(eval_ood_KNN(ID_training_data, ID_val_data, OOD_data, k=knn_k))
# seems to be some slight numerical differences here


