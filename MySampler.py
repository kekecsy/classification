from torch.utils.data import Sampler
import random
import numpy as np

class ProgressiveLabelSampler(Sampler):
    def __init__(self, data_source, high_freq_label, low_freq_label, max_epochs, start_fraction=0.1):
        """
        Args:
            data_source (Dataset): 数据集
            target_labels (list): 需要逐渐增加比例的标签列表
            max_epochs (int): 最大 epoch 数
            start_fraction (float): 起始数据比例
        """
        self.data_source = data_source
        self.high_freq_label = set(high_freq_label.keys())
        self.low_freq_label = set(low_freq_label.keys())
        self.max_epochs = max_epochs
        self.start_fraction = start_fraction
        self.current_epoch = 0
        self.high_freq_label_indices = self.data_source.high_freq_label_indices
        self.low_freq_label_indices = self.data_source.low_freq_label_indices
        self.other_indices = self.data_source.other_indices
        # self.high_freq_label_dict = self.data_source.high_freq_label_dict


    def set_epoch(self, epoch):
        """在每个 epoch 开始时设置当前 epoch"""
        self.current_epoch = epoch

    def __iter__(self):
        # 计算当前标签数据的比例，逐渐增加到 100%
        highfreq_fraction = min(self.start_fraction + (1 - self.start_fraction) * (self.current_epoch / self.max_epochs), 1.0)

        # lowfreq_fraction = min(3 + self.start_fraction + (1 - self.start_fraction) * (self.current_epoch / self.max_epochs), 5.0)

        # 根据比例选择目标标签的样本
        indices = []

        # 根据高频标签，按照比例采样
        # for _, label_idx in self.high_freq_label_dict.items():
        #     num_samples = int(len(label_idx) * highfreq_fraction)
        #     indices.extend(random.sample(label_idx, num_samples))

        # for _, label_idx in self.low_freq_label_indices.items():
        #     num_samples = int(len(label_idx) * lowfreq_fraction)   # 这里要多倍数采样，还没写好
        #     indices.extend(random.sample(label_idx, num_samples))

        # 直接在所有高频样本中采样
        indices.extend(random.sample(self.high_freq_label_indices, 
                                     int(len(self.high_freq_label_indices) * highfreq_fraction)))

        # indices.extend(random.choices(self.low_freq_label_indices, 
        #                              k=int(len(self.low_freq_label_indices) * lowfreq_fraction)))

        # 直接添加低频样本
        indices.extend(self.low_freq_label_indices)

        # 添加其他标签的样本
        indices.extend(self.other_indices)

        indices = list(set(indices))    # 一定要去重，不然会多进程访问相同数据，产生冲突

        # 随机打乱采样顺序
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        # 返回每个 epoch 中采样的数据量
        highfreq_fraction = min(self.start_fraction + (1 - self.start_fraction) * (self.current_epoch / self.max_epochs), 1.0)
        return len(self.other_indices) + len(self.low_freq_label_indices) + int(len(self.high_freq_label_indices) * highfreq_fraction)