import torchmetrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from lightning.pytorch.callbacks import Callback
from .save_knn_pred import KNNPrediction

from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import copy

from pdb import set_trace as pb
from tqdm import tqdm

class PAWSPrediction(KNNPrediction):

    def __init__(self, backbone='backbone', check_every: int = 10,
            temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.backbone_str = backbone
        self.check_every = check_every
        self.dp = nn.Parameter(torch.tensor(0.0))
        self.reported = True
        self.softmax = torch.nn.Softmax(dim=1)

        self.labelled_indices = None

    def setup(self, trainer, pl_module, stage):
        super().setup(trainer, pl_module, stage)
        self.labelled_indices = pl_module.labelled_indices

    def setup_data(self, trainer, stage='val', basedata='train', **kwargs):
        return self._setup_data(trainer, stage, self.labelled_indices, basedata=basedata, **kwargs)

    def evaluate_batch(self, feature, train_feature_bank_all, train_targets_bank_all, return_likelihood=False):
        feature = feature.squeeze()
        with torch.cuda.amp.autocast(enabled=False):
            feature, train_feature_bank_all = feature.float(), train_feature_bank_all.float()
            feature = F.normalize(feature, dim=1)
            
            exponent = feature @ train_feature_bank_all /self.temperature
            pred_labels = self.softmax(exponent) @ train_targets_bank_all
            likelihood = torch.max(exponent, dim=(1)).values

        if return_likelihood:
            return pred_labels, likelihood
        else:
            return pred_labels

    def evaluate(self, trainer, pl_module, stage='val', save_predictions=False):
        train_data, test_data = self.setup_data(trainer, stage)
        if dist.is_initialized() and dist.get_world_size() > 1:
            rank = dist.get_rank()
        else:
            rank = 0
        train_feature_bank_all, train_targets_bank_all = self._evaluate_embeddings(train_data, trainer, rank)
        # ====================================================
        num = torch.zeros([self.num_classes]).to(self.dp.device)
        top1 = torch.zeros([self.num_classes]).to(self.dp.device)
        train_targets_bank_all = torch.nn.functional.one_hot(train_targets_bank_all).to(train_feature_bank_all)
        predictions = []
        indices = []
        with torch.no_grad():
            for i,data in enumerate(tqdm(test_data, position=rank, miniters=self.miniters)):
                [img], target, idx = data
                img, target = img.to(self.dp.device), target.to(self.dp.device)
                feature = self.backbone(img)
                pred_labels = self.evaluate_batch(feature, train_feature_bank_all, train_targets_bank_all)
                if save_predictions:
                    predictions.append(pred_labels)
                    indices.append(np.array(idx, dtype=int)[:,0])
                    
                unq = torch.unique(target, return_counts=True)
                num[unq[0]] += unq[1]
                pred_correct = (pred_labels.argmax(dim=1) == target)
                unq_correct = torch.unique(target[pred_correct], return_counts=True)
                top1[unq_correct[0]] += unq_correct[1]
                if self.val_limit_batches:
                    if i == trainer.limit_val_batches:
                        break

        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(num)
            dist.all_reduce(top1)

        acc = float((top1.sum() / num.sum()).item())
        mean_acc = float((top1 / num).nanmean())
        self.report_acc = acc
        self.report_acc_mean = mean_acc
        self.reported = False
        if save_predictions:
            predictions = torch.cat(predictions, dim=0)
            indices = np.concatenate(indices)
            return predictions, indices

# ==================================================

    def on_validation_epoch_end(self, trainer, pl_module):
        self._on_epoch_end(trainer, pl_module, string='val/PAWS_accuracy')
    def on_test_epoch_end(self, trainer, pl_module):
        self._on_epoch_end(trainer, pl_module, string='test/PAWS_accuracy')