import os
import time
import torch
import pandas as pd
import numpy as np
from torch.optim import Adam
from accelerate import Accelerator
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Sampler, BatchSampler
from transformers import T5EncoderModel, T5Tokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, multilabel_confusion_matrix
from ProtTransModel2 import ClassConfig, T5EncoderCLSModel, Contrastive_loss, LayerLoss, NodeLoss
from accelerate.logging import get_logger
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
import json
import math
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from MySampler import CommonLabelSampler, CycleLabelSampler, SelectBatchSampler
import torch.distributed as dist
import torch.multiprocessing as mp



model_dirname = '/step_Focalloss6_'
model_path = '/data/csyData/prot_t5_xl_half_uniref50-enc'

with open('/data/csyData/uniprot_test/data/XML_testdata/go_temp/GO_data/cco/label_freq_list.json') as f:
    label_freq_list = json.load(f)

path = '/data/csyData/uniprot_test/data/XML_testdata/go_temp/GO_data/cco/withlayer2/'
datatype = 'stride/'
train_data_dir1 = path + datatype + 'traindata.csv'
train_data_dir2 = path + datatype + 'trainNolabel.csv'
# train_data_dir3 = path + 'traindata2.csv'
train_filelist = [train_data_dir1, train_data_dir2]


test_data_dir1 = path + datatype + 'testdata.csv'
test_data_dir2 = path + datatype + 'testNolabel.csv'
# test_data_dir3 = path + 'testdata2.csv'
test_filelist = [test_data_dir1, test_data_dir2]

# 得到 根 -> 叶子 字典
with open('/data/csyData/uniprot_test/data/XML_testdata/go_temp/GO_data/cco/GO_cco_withlayerDepthDict3.json') as f:
    temp_hierar_relations = json.load(f)
    for go in temp_hierar_relations:
        temp_hierar_relations[go] = temp_hierar_relations[go]['leaf']

# 得到 分层聚类字典
with open('/data/csyData/pygosemsim-master/pygosemsim-master/cco_Kmeans_cluster2.json') as f:
    temp_cluster_nodes = json.load(f)

# 得到 聚类的父子关系字典
# with open('/data/csyData/uniprot_test/code/GOcode/cco_version2/cco_cluster_relations.json') as f:
with open('/data/csyData/pygosemsim-master/pygosemsim-master/cco_KMeanscluster_relations.json') as f:
    cluster_relations = json.load(f)

# 得到标签频率按照阈值分成四份的字典
# with open('/data/csyData/uniprot_test/data/XML_testdata/go_temp/GO_data/cco/label_freq_list.json') as f:
#     label_freq_list = json.load(f)

# 得到标签频率按照字典
# with open('/data/csyData/uniprot_test/data/XML_testdata/go_temp/GO_data/cco/sta_count3.json') as f:
#     sta_count3 = json.load(f)

# 得到alpha
with open('/data/csyData/uniprot_test/code/GOcode/cco_version2/freq.json') as f:
    freq = json.load(f)

node_nums = len(temp_hierar_relations.keys()) - 1
layer_nums = len(temp_cluster_nodes.keys())
labels_num = node_nums + layer_nums

epoch = 200    # 可能修改
max_length = 512
train_batch_size = 64
valid_batch_size = 64
# lr = 3e-4
# lr = 1e-3
only_layer_metrics = False
# load_model = True
# saved_ep = 50   # 重新训练时，记得改成最新的模型参数
saved_ep = 0

temp_go = list(pd.read_csv(test_data_dir2, nrows=0).columns)[2:]

high_freq = {}
low_freq = {}
hierar_relations = {}

def go2index(target_go):
    go = temp_go
    if isinstance(target_go, list):
        return [go.index(i) for i in target_go]
    elif isinstance(target_go, str):
        if target_go in go:
            return go.index(target_go)
        else:
            return None
        
for go,value in freq.items():
    # alpha[go2index(go)] = value
    if value < 0.5:
        high_freq[go] = value
    elif value > 0.94:
        low_freq[go] = value

for node,value in temp_hierar_relations.items():
    if go2index(node):
        hierar_relations[str(go2index(node))] = go2index(value)

main_numbers, sub_numbers = [], []
for label in list(temp_cluster_nodes.keys()):
    if '_' not in label:
        main_numbers.append(label)
    else:
        sub_numbers.append(label)

main_numbers.sort(key=lambda x: int(x))
sub_numbers.sort(key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1])) if '_' in x else (int(x), -1))
cluster_nodes = main_numbers + sub_numbers


cluster_nodes_relations = {}
for index,value in enumerate(cluster_nodes):
    cluster_nodes_relations[str(index)] = go2index(temp_cluster_nodes[value])

alpha = [0] * node_nums

model = T5EncoderModel.from_pretrained(model_path)
model.config.d_model = 1024
tokenizer = T5Tokenizer.from_pretrained(model_path)

lora_target_modules = []
for name, _ in model.named_modules():
    for layer in [23]:
        if (str(layer) in name) & (('.wi' in name) or ('.wo' in name) or ('.q' in name) or ('.k' in name) or ('.v' in name) or ('.o' in name)):
            lora_target_modules.append(name)

peft_config = LoraConfig(
                lora_alpha=16,
                lora_dropout=0.1,
                r=8,
                bias="none",
                task_type="SEQ_2_SEQ_LM",
                target_modules=lora_target_modules
                )
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

class_config = ClassConfig(node_nums=node_nums, layer_nums=layer_nums, batch_size=train_batch_size)

class_model = T5EncoderCLSModel(model.config,
                                class_config,
                                cluster_relations,
                                hierar_relations,
                                cluster_nodes_relations,
                                alpha)
class_model.encoder = model.encoder

model = class_model
del class_model
model = model.cuda()

from safetensors.torch import load_file

saved_path = './ckpts6/step_1600/model/model.safetensors'

weights = load_file(saved_path)

model.load_state_dict(weights)


def collate_func(batch):
    sequence, layers, nodes = [], [], []
    for item in batch:
        sequence.append(item[0])
        layers.append(item[1])
        nodes.append(item[2])
    inputs = tokenizer(sequence, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    # del inputs["token_type_ids"]
    if not only_layer_metrics:
        inputs["nodes"] = torch.FloatTensor(np.stack(nodes)).to('cuda:1')
    inputs["layers"] = torch.FloatTensor(np.stack(layers)).to('cuda:1')
    return inputs

class MyDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(MyDataLoader, self).__init__(*args, **kwargs)
        self.go = self.dataset.go

    def go2index(self,target_go):
        if isinstance(target_go, list):
            return [self.go.index(i) for i in target_go]
        elif isinstance(target_go, str):
            if target_go in self.go:
                return self.go.index(target_go)
            else:
                return None

class MyDataset(Dataset):

    def __init__(self,data_dirlist) -> None:
        super().__init__()
        self.data = pd.concat(data_dirlist,axis=0).reset_index(drop=True)
        self.go = list(self.data.columns)[2:]
        
        # low_freq_label_indices = self.data[self.data[list(low_freq.keys())].eq(1).any(axis=1)].index.tolist()
        # temp = self.data.loc[low_freq_label_indices]
        # temp['sequence'] = temp['sequence'].apply(lambda x: x[::-1])
        # self.data = pd.concat([self.data] + [temp]).reset_index(drop=True)

        self.high_freq_label_indices = self.data[self.data[list(high_freq.keys())].eq(1).any(axis=1) & self.data[list(low_freq.keys())].eq(0).all(axis=1)].index
        self.low_freq_label_indices = self.data[self.data[list(low_freq.keys())].eq(1).any(axis=1)].index.difference(self.high_freq_label_indices)

        self.other_indices = self.data.index.difference(self.low_freq_label_indices).difference(self.high_freq_label_indices)

    def __getitem__(self, index):
        temp_nodelabel = np.array(self.data.iloc[index,2:].tolist())
        temp_layerlabel = np.array([])
        
        for cluster_name in cluster_nodes:
            golist = temp_cluster_nodes[cluster_name]
            temp_layerlabel = np.append(temp_layerlabel, 
                                        np.sum(temp_nodelabel[np.array(self.go2index(golist))]) > 0)
        return self.data["sequence"].iloc[index], temp_layerlabel, temp_nodelabel
    
    def __len__(self):
        return len(self.data)

    def go2index(self,target_go):
        go = self.go
        if isinstance(target_go, list):
            return [go.index(i) for i in target_go]
        elif isinstance(target_go, str):
            if target_go in go:
                return go.index(target_go)
            else:
                return None


validdata_list = []
for i in test_filelist:
    validdata_list.append(pd.read_csv(i))
validset = MyDataset(validdata_list)

# dist.init_process_group(backend='nccl', init_method='tcp://localhost:29500',
#                             world_size=1, rank=2)

accelerator = Accelerator(gradient_accumulation_steps=8, log_with="tensorboard", project_dir="ckpts0")
accelerator.init_trackers("runs")

high_freq_sampler = CommonLabelSampler(validset.high_freq_label_indices[:132], \
                                    num_replicas=accelerator.num_processes, \
                                    rank=accelerator.process_index, \
                                    shuffle=True, \
                                    drop_last=True)

low_freq_sampler = CycleLabelSampler(validset.low_freq_label_indices, \
                                    num_replicas=accelerator.num_processes, \
                                    rank=accelerator.process_index, \
                                    shuffle=True, \
                                    drop_last=False)

other_sampler = CycleLabelSampler(validset.other_indices, \
                                    num_replicas=accelerator.num_processes, \
                                    rank=accelerator.process_index, \
                                    shuffle=True, \
                                    drop_last=False)

valid_batch_sampler = SelectBatchSampler(high_freq_sampler, 
                                         low_freq_sampler,
                                         other_sampler,
                                         batch_size=valid_batch_size,
                                         drop_last=True)


validloader = MyDataLoader(validset, 
                           collate_fn=collate_func, 
                           batch_sampler=valid_batch_sampler)

model, validloader = accelerator.prepare(model, validloader)

count = 0
for idx, i in enumerate(validloader):
    # print(i)
    count += 1
    print(i['input_ids'].size())

print(count)

