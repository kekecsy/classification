from torch.utils.data import Sampler, BatchSampler
import random
import torch
import math
import itertools

class SelectBatchSampler(BatchSampler):
    def __init__(self, high_sampler, low_sampler, other_sampler, batch_size, drop_last, custom_logic=None):
        self.high_sampler = high_sampler
        self.low_sampler = low_sampler
        self.other_sampler = other_sampler

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.custom_logic = custom_logic if custom_logic is not None else lambda x: x

    def __iter__(self):
        high_iter = self.high_sampler
        low_cycle_iter = self.low_sampler
        other_cycle_iter = self.other_sampler
        all_batch = []
        for i in high_iter:
            all_batch.append(i)
            if len(all_batch) == self.batch_size // 2:
                for j in low_cycle_iter:
                    all_batch.append(j)
                    if len(all_batch) == self.batch_size * 3 // 4:
                        break
                for k in other_cycle_iter:
                    all_batch.append(k)
                    if len(all_batch) == self.batch_size:
                        break
                yield self.custom_logic(all_batch)
                all_batch = []
        if len(all_batch) > 0 and not self.drop_last:
            yield self.custom_logic(all_batch)

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size



class CommonLabelSampler(Sampler):
    def __init__(self, indices_data, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False):
        self.data_source = indices_data
        self.num_replicas = num_replicas or torch.distributed.get_world_size()
        self.rank = rank or torch.distributed.get_rank()
        self.total_size = len(self.data_source)
        self.shuffle = shuffle
        self.drop_last = drop_last
        if self.drop_last:
            self.num_samples = len(self.data_source) // self.num_replicas
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.num_samples = math.ceil(len(self.data_source) / self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = self.data_source.tolist()
        if self.shuffle:
            random.shuffle(indices)
        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size
        indices_per_process = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices_per_process)

    def __len__(self):
        return self.num_samples
    

class CycleLabelSampler(Sampler):
    def __init__(self, indices_data, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False):
        self.data_source = indices_data
        self.num_replicas = num_replicas or torch.distributed.get_world_size()
        self.rank = rank or torch.distributed.get_rank()
        self.total_size = len(self.data_source)
        self.shuffle = shuffle
        self.drop_last = drop_last
        if self.drop_last:
            self.num_samples = self.total_size // self.num_replicas
        else:
            self.num_samples = math.ceil(self.total_size / self.num_replicas)
        
    def __iter__(self):
        indices = self.data_source.tolist()
        if self.shuffle:
            random.shuffle(indices)
        indices_per_process = itertools.cycle(indices[self.rank:self.total_size:self.num_replicas])
        return iter(indices_per_process)

    def __len__(self):
        return self.num_samples