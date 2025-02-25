from PIL import ImageFilter, ImageOps
import random
from torchvision import transforms
from PIL import Image
from lightly.transforms.multi_view_transform import MultiViewTransform
import numpy as np
import torch

from .paws_helpers import Solarize, Equalize, GaussianBlur

from pdb import set_trace as pb

def get_color_distortion(s=1.0, grayscale = False, solarize = True, equalize = True):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)

    base_distort = [rnd_color_jitter]
    if grayscale:
        base_distort.append(rnd_gray)
    if solarize:
        base_distort.append(Solarize(p=0.2))
    if equalize:
        base_distort.append(Equalize(p=0.2))

    color_distort = transforms.Compose(base_distort)
    return color_distort

class DataAugmentationPAWS(MultiViewTransform):
    def __init__(self, 
        global_crops_scale=(0.75, 1.0), 
        local_crops_scale=(0.3, 0.75), 
        global_crops_number=2, 
        local_crops_number=8, 
        global_crop_size=32,
        local_crop_size=18,
        color_distortion_s = 0.5,
        color_distortion_grayscale = False,
        color_distortion_solarize = True,
        color_distortion_equalize = True,
        blur_prob = 0.0,
        normalize = {'mean': (0.4914, 0.4822, 0.4465), 'std': (0.2023, 0.1994, 0.2010)},
        static_seed=None, **kwargs):
        self.static_seed = static_seed

        global_transform = [
             transforms.RandomResizedCrop(size=global_crop_size, scale=global_crops_scale,
                    interpolation=transforms.InterpolationMode.BILINEAR),
             transforms.RandomHorizontalFlip(),
             get_color_distortion(color_distortion_s, color_distortion_grayscale, 
                    color_distortion_solarize, color_distortion_equalize),
             GaussianBlur(p=blur_prob),
             transforms.ToTensor(),
        ]

        local_tranform = [
             transforms.RandomResizedCrop(size=local_crop_size, scale=local_crops_scale,
                    interpolation=transforms.InterpolationMode.BILINEAR),
             transforms.RandomHorizontalFlip(),
             get_color_distortion(color_distortion_s, color_distortion_grayscale, 
                    color_distortion_solarize, color_distortion_equalize),
             GaussianBlur(p=blur_prob),
             transforms.ToTensor(),
        ]

        if normalize:
            global_transform += [transforms.Normalize(mean=normalize["mean"], std=normalize["std"])]
            local_tranform += [transforms.Normalize(mean=normalize["mean"], std=normalize["std"])]

        global_transform = transforms.Compose(global_transform)
        local_tranform = transforms.Compose(local_tranform)

        transform_list=[global_transform for x in range(global_crops_number)]
        for _ in range(local_crops_number):
            transform_list.append(local_tranform)

        super().__init__(transforms=transform_list)

    def __call__(self, image):
        if self.static_seed is not None:
            torch.manual_seed(self.static_seed)
            np.random.seed(self.static_seed)
            random.seed(self.static_seed)
            torch.cuda.manual_seed_all(self.static_seed)

        t = [transform(image) for transform in self.transforms]
        return t