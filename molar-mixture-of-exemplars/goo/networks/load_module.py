
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as pb
from ..utils.utils import activate_requires_grad

from typing import Any, Mapping
from lightning import LightningModule
from os.path import exists
import re

class LoadModule(LightningModule):
    def __init__(self, model_class, model_path, num_classes=None, 
        module_name=None, model_hparams = None, seed = -1,
        activate_grad = False, set_attributes = {}, **kwargs): #kwargs as extra stuff gets kept
        super().__init__()

        if "XXXXX" in model_path:
            model_path = re.sub('XXXXX', str(seed), model_path)

        if model_hparams is None:
            if num_classes is None:
                self.model = model_class.func.load_from_checkpoint(model_path)
            else:
                self.model = model_class.func.load_from_checkpoint(model_path, num_classes=num_classes)
            model_hparams = self.model.hparams
        else:
            self.model = model_class(**model_hparams)
        self.save_hyperparameters()
        self.model = getattr(self.model, self.hparams.module_name)
        if activate_grad:
            activate_requires_grad(self.model)
        for key in set_attributes.keys():
            setattr(self.model, key, set_attributes[key])

# ==============================================================
    def forward(self, x):
        embed = self.model(x)
        return embed
