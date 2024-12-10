import math
import torch
from torch.utils.data import Sampler
import random
from torch.utils.data.sampler import Sampler, BatchSampler
import itertools


class MyDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False):
        self.dataset = dataset
        self.num_replicas = num_replicas or torch.distributed.get_world_size()
        self.rank = rank or torch.distributed.get_rank()
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        # self.ratio = initial_ratio  # 初始比例

        # 初次计算样本数，考虑 drop_last
        # self.update_num_samples()

    # def update_num_samples(self):
    #     """更新当前的 num_samples，根据 drop_last 和 ratio 来调整样本数量"""

    #     self.high_freq_label_indices = self.dataset.high_freq_label_indices
    #     self.low_freq_label_indices = self.dataset.low_freq_label_indices
    #     self.other_indices = self.dataset.other_indices
    #     # temp = 0
    #     # for key in self.dataset.high_freq_label_dict:
    #     #     temp += int(len(self.dataset.high_freq_label_dict[key]) * self.ratio)
    #     # total_size = int(len(self.dataset.high_freq_label_indices) * self.ratio) + len(self.low_freq_label_indices) + len(self.other_indices)
    #     total_size = len(self.high_freq_label_indices) + len(self.low_freq_label_indices) + len(self.other_indices)

    #     if self.drop_last:
    #         self.num_samples = total_size // self.num_replicas
    #     else:
    #         self.num_samples = math.ceil(total_size / self.num_replicas)
    
    #     # 确保总样本量符合 DataLoader 需求
    #     self.total_size = self.num_samples * self.num_replicas

    # def set_epoch(self, epoch):
    #     """根据 epoch 更新随机种子和比例，并重新计算样本数"""
    #     self.seed = epoch
    #     self.ratio = min(1.0, self.ratio + 0.1 * epoch)  # 逐渐增加 ratio
    #     self.update_num_samples()  # 更新样本数量

    def __iter__(self):
        # 生成全部数据索引
        # indices = list(range(len(self.dataset)))
        
        # 根据比例选择一部分数据（例如，取前 self.ratio 比例的数据）
        # boundary_index = int(len(indices) * self.ratio)
        # selected_indices = random.sample(indices, boundary_index)

        # selected_indices = []
        # selected_indices.extend(random.sample(self.high_freq_label_indices, 
        #                         int(len(self.high_freq_label_indices) * self.ratio)))

        # selected_indices.extend(self.low_freq_label_indices)
        # selected_indices.extend(self.other_indices)
        high_freq_indices = self.high_freq_label_indices.tolist()
        low_freq_indices = self.low_freq_label_indices.tolist()
        other_indices = self.other_indices.tolist()

        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            high_freq_indices = torch.randperm(len(high_freq_indices), generator=generator).tolist()
            low_freq_indices = torch.randperm(len(low_freq_indices), generator=generator).tolist()
            other_indices = torch.randperm(len(other_indices), generator=generator).tolist()


        # 按照 rank 选择当前 GPU 的样本
        high_freq_indices_per_process = high_freq_indices[self.rank:self.total_size:self.num_replicas]
        # low_freq_indices_per_process = low_freq_indices[self.rank:self.total_size:self.num_replicas]
        # other_indices_per_process = other_indices[self.rank:self.total_size:self.num_replicas]
        low_freq_indices_per_process = itertools.cycle(low_freq_indices[self.rank:self.total_size:self.num_replicas])
        other_indices_per_process = itertools.cycle(other_indices[self.rank:self.total_size:self.num_replicas])
        
        return (iter(high_freq_indices_per_process),
                iter(low_freq_indices_per_process),
                iter(other_indices_per_process)
                )

    def __len__(self):
        """返回当前比例下的采样长度，符合 DataLoader 要求"""
        return self.num_samples