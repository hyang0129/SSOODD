import torch
import torch.distributed as dist
import numpy as np
from pdb import set_trace as pb
import math
from .dummy_sampler import DummySampler
# https://github.com/facebookresearch/suncet/blob/main/src/data_manager.py

def set_target_indices(dataset): # assumes data source is lightly wrapper with mysubset dataset object
    # Must be a dataset that is a subset to get the labelled indices
    assert(hasattr(dataset.dataset, 'indices'))
    labelled_indices = dataset.dataset.indices
    dataset.dataset.target_indices = []
    effective_classes = 0
    for t in dataset.dataset.classes:
        indices = np.squeeze(np.argwhere(dataset.dataset.targets == t)).tolist()
        if not isinstance(indices, list): # this takes care of instances when only one index for a class. It will still fail if there are not single class instances
            indices = [indices]
        if len(indices) > 0:
            # indices = labelled_indices[indices]
            dataset.dataset.target_indices.append(indices)
            effective_classes += 1
    return effective_classes

class ClassStratifiedSampler(torch.utils.data.Sampler):

    def __init__(
        self,
        dataset,
        batch_size=1,
        world_size = None,
        rank = None,
        classes_per_batch=10,
        resample_method='stratified', # 'stratified'
        seed=0,
        unique_classes=False,
        epochs=1
    ):
        """
        ClassStratifiedSampler

        Batch-sampler that samples 'batch-size' images from subset of randomly
        chosen classes e.g., if classes a,b,c are randomly sampled,
        the sampler returns
            torch.cat([a,b,c], [a,b,c], ..., [a,b,c], dim=0)
        where a,b,c, are images from classes a,b,c respectively.
        Sampler, samples images WITH REPLACEMENT (i.e., not epoch-based)

        :param dataset: dataset of type "TransImageNet" or "TransCIFAR10'
        :param world_size: total number of workers in network
        :param rank: local rank in network
        :param batch_size: num. images to load from each class
        :param classes_per_batch: num. classes to randomly sample for batch
        :param epochs: num consecutive epochs thru data_source before gen.reset
        :param seed: common seed across workers for subsampling classes
        :param unique_classes: true ==> each worker samples a distinct set of classes; false ==> all workers sample the same classes
        """
        super(ClassStratifiedSampler, self).__init__(dataset)
        self.dataset = dataset
        self.resample_method = resample_method

        if torch.distributed.is_initialized():
            if world_size is None:
                if not dist.is_available():
                    raise RuntimeError("Requires distributed package to be available")
                world_size = dist.get_world_size()
            if rank is None:
                if not dist.is_available():
                    raise RuntimeError("Requires distributed package to be available")
                rank = dist.get_rank()
            if rank >= world_size or rank < 0:
                raise ValueError(
                    "Invalid rank {}, rank should be in the interval"
                    " [0, {}]".format(rank, num_replicas - 1))
        else:
            rank = 0
            world_size=1

        self.rank = rank
        self.world_size = world_size
        self.unique_cpb = unique_classes
        self.effective_classes = set_target_indices(dataset)
        self.num_classes = self.effective_classes
        self.epochs = epochs

        if self.unique_cpb:
            self.cpb = min(classes_per_batch, self.num_classes // world_size)
        else:
            self.cpb = min(classes_per_batch, self.num_classes)

        self.batch_size = max(batch_size//(self.cpb*self.world_size), 1) # batch size per class
        self.batches_per_iteration = len(self.dataset)//(self.batch_size*self.cpb*self.world_size)

        # self.batch_size = max(batch_size//self.num_classes, 1)
        # self.batches_per_iteration = len(self.dataset)//(self.batch_size*self.num_classes)

        self.batches_per_iteration = max(self.batches_per_iteration, 1)

        if not self.unique_cpb:
            assert self.num_classes % self.cpb == 0

        self.base_seed = seed  # instance seed
        self.seed = seed  # subsample sampler seed

        # self.outer_epoch = 0
        self.sampler = DummySampler()

    @property
    def outer_epoch(self):
        return self.sampler.epoch
    # def set_epoch(self, epoch):
    #     self.outer_epoch = epoch

    def set_inner_epochs(self, epochs):
        self.epochs = epochs

    def _next_perm(self):
        self.seed += 1
        g = torch.Generator()
        g.manual_seed(self.seed)
        self._perm = torch.randperm(self.num_classes, generator=g)

    def _get_perm_ssi(self):
        start = self._ssi
        end = self._ssi + self.cpb
        subsample = self._perm[start:end]
        return subsample

    def _next_ssi(self):
        if not self.unique_cpb:
            self._ssi = (self._ssi + self.cpb) % self.num_classes
            if self._ssi == 0:
                self._next_perm()
        else:
            self._ssi += self.cpb * self.world_size
            max_end = self._ssi + self.cpb * (self.world_size - self.rank)
            if max_end > self.num_classes:
                self._ssi = self.rank * self.cpb
                self._next_perm()

    def _get_local_samplers(self, epoch):
        """ Generate samplers for local data set in given epoch """
        seed = int(self.base_seed + epoch
                   + self.epochs * self.rank
                   + self.outer_epoch * self.epochs * self.world_size)
        g = torch.Generator()
        g.manual_seed(seed)
        samplers = []
        for t in range(self.num_classes):
            t_indices = np.array(self.dataset.dataset.target_indices[t])
            if not self.unique_cpb:
                i_size = len(t_indices) // self.world_size
                if i_size > 0:
                    t_indices = t_indices[self.rank*i_size:(self.rank+1)*i_size]
            if len(t_indices) > 1:
                t_indices = t_indices[torch.randperm(len(t_indices), generator=g)]

            # If amount of data is less than batch size, resample images!
            if len(t_indices) < self.batch_size:
                if self.resample_method == 'random':
                    t_indices = t_indices[torch.randint(0, t_indices.shape[0], (self.batch_size,), generator=g)]
                elif self.resample_method == 'stratified':
                    num_expand_x = math.ceil(self.batch_size / len(t_indices))
                    raw_indices = np.hstack([torch.randperm(len(t_indices), generator=g) for _ in range(num_expand_x)])
                    t_indices = t_indices[raw_indices]
            samplers.append(iter(t_indices))
        return samplers

    def _subsample_samplers(self, samplers):
        """ Subsample a small set of samplers from all class-samplers """
        subsample = self._get_perm_ssi()
        subsampled_samplers = []
        for i in subsample:
            subsampled_samplers.append(samplers[i])
        self._next_ssi()
        return zip(*subsampled_samplers)

    def __iter__(self):
        self._ssi = self.rank*self.cpb if self.unique_cpb else 0
        self._next_perm()

        # -- iterations per epoch (extract batch-size samples from each class)
        ipe = (self.num_classes // self.cpb if not self.unique_cpb
               else self.num_classes // (self.cpb * self.world_size)) * self.batch_size

        for epoch in range(self.epochs):

            # -- shuffle class order
            samplers = self._get_local_samplers(epoch)
            subsampled_samplers = self._subsample_samplers(samplers)

            counter, batch = 0, []
            for i in range(ipe):
                batch += list(next(subsampled_samplers))
                counter += 1
                if counter == self.batch_size:
                    yield batch
                    counter, batch = 0, []
                    if i + 1 < ipe:
                        subsampled_samplers = self._subsample_samplers(samplers)

    def __len__(self):
        if self.batch_size == 0:
            return 0

        ipe = (self.num_classes // self.cpb if not self.unique_cpb
               else self.num_classes // (self.cpb * self.world_size))
        return self.epochs * ipe


# class ClassStratifiedSamplerWrapper(torch.utils.data.Sampler):

#     def __init__(
#         self,
#         data_source,
#         batch_size=1,
#         world_size = None,
#         rank = None,
#         classes_per_batch=10,
#         epochs=1,
#         seed=0,
#         unique_classes=False
#     ):
#         super(ClassStratifiedSampler, self).__init__(data_source)
#         self.sampler = ClassStratifiedSampler(data_source, batch_size, world_size, rank, classes_per_batch, epochs, seed, unique_classes)

#     def __iter__(self):


#     def __len__(self):