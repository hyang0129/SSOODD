
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from collections import OrderedDict
from pdb import set_trace as pb

class ProjectionHead(nn.Module):
    def __init__(self, hidden_dim=128, output_dim=128, init_seed=-1, 
            batch_norm = True, last_layer_bias=True, **kwargs):
        super(ProjectionHead, self).__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        if init_seed != -1:
            torch.manual_seed(init_seed)
            np.random.seed(init_seed)
            random.seed(init_seed)
            torch.cuda.manual_seed_all(init_seed)

        if batch_norm:
            self.fc = torch.nn.Sequential(OrderedDict([
                ('fc1', torch.nn.Linear(hidden_dim, hidden_dim)),
                ('bn1', torch.nn.BatchNorm1d(hidden_dim)),
                ('relu1', torch.nn.ReLU(inplace=True)),
                ('fc2', torch.nn.Linear(hidden_dim, hidden_dim)),
                ('bn2', torch.nn.BatchNorm1d(hidden_dim)),
                ('relu2', torch.nn.ReLU(inplace=True)),
                ('fc3', torch.nn.Linear(hidden_dim, output_dim, bias=last_layer_bias))
            ]))
        else:
            self.fc = torch.nn.Sequential(OrderedDict([
                ('fc1', torch.nn.Linear(hidden_dim, hidden_dim)),
                ('relu1', torch.nn.ReLU(inplace=True)),
                ('fc2', torch.nn.Linear(hidden_dim, hidden_dim)),
                ('relu2', torch.nn.ReLU(inplace=True)),
                ('fc3', torch.nn.Linear(hidden_dim, output_dim, bias=last_layer_bias))
            ]))
        # self.fc_final = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, **kwargs):
        x = self.fc(x)
        # x = self.fc_final(x)
        return x

class PredictionHead(nn.Module):
    def __init__(self, hidden_dim=128, output_dim=128):
        super(PredictionHead, self).__init__()
        pred_head = OrderedDict([])
        pred_head['bn1'] = torch.nn.BatchNorm1d(output_dim)
        pred_head['fc1'] = torch.nn.Linear(output_dim, output_dim//mx)
        pred_head['bn2'] = torch.nn.BatchNorm1d(output_dim//mx)
        pred_head['relu'] = torch.nn.ReLU(inplace=True)
        pred_head['fc2'] = torch.nn.Linear(output_dim//mx, output_dim)
        self.pred = torch.nn.Sequential(pred_head)

    def forward(self, x, **kwargs):
        x = self.pred(x)
        return x

# ==========================================


class ProjectionHeadKappa(nn.Module):
    def __init__(self, hidden_dim=128, output_dim=128, init_seed=-1, **kwargs):
        super(ProjectionHead, self).__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        if init_seed != -1:
            torch.manual_seed(init_seed)
            np.random.seed(init_seed)
            random.seed(init_seed)
            torch.cuda.manual_seed_all(init_seed)

        self.fc_dir = torch.nn.Sequential(OrderedDict([
            ('fc1', torch.nn.Linear(hidden_dim, hidden_dim, bias=False)),
            ('bn1', torch.nn.BatchNorm1d(hidden_dim)),
            ('relu1', torch.nn.ReLU(inplace=True)),
            ('fc2', torch.nn.Linear(hidden_dim, hidden_dim, bias=False)),
            ('bn2', torch.nn.BatchNorm1d(hidden_dim)),
            ('relu2', torch.nn.ReLU(inplace=True)),
            ('fc3', torch.nn.Linear(hidden_dim, output_dim, bias=False))
        ]))
        # self.fc_final = torch.nn.Linear(hidden_dim, output_dim)
        self.fc_rad = torch.nn.Sequential(OrderedDict([
            ('fc1', torch.nn.Linear(hidden_dim, hidden_dim, bias=True)),
            ('bn1', torch.nn.BatchNorm1d(hidden_dim)),
            ('relu1', torch.nn.ReLU(inplace=True)),
            ('fc2', torch.nn.Linear(hidden_dim, 1, bias=True))
        ]))

    def forward(self, x, **kwargs):
        x_dir = self.fc(x)
        x_dir = torch.nn.functional.normalize(x_dir, dim=1)
        # x = self.fc_final(x)
        pb()
        return x


