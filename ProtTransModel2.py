import copy
import numpy as np
import random
import json

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from evaluate import load
import math
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.t5.modeling_t5 import T5Config, T5PreTrainedModel, T5Stack
from transformers import BertModel, BertTokenizer, BertConfig
from transformers import TrainingArguments, Trainer, set_seed
from loss import ClassificationLoss
from loss import LossType
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.utils import ModelOutput


# 设置种子，保证实验可再现
def set_seeds(s):
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    set_seed(s)


def compute_loss(score,y_true):
    return torch.log(1 + torch.sum(torch.exp(torch.mul(y_true,score)).masked_fill(torch.mul(y_true,score)==0,0))) \
    + torch.log(1 + torch.sum(torch.exp(torch.mul(1 - y_true,score)).masked_fill(torch.mul(1 - y_true,score)==0,0)))


def multilabel_categorical_crossentropy(y_pred,y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.concatenate([y_pred_neg, zeros], axis=-1)
    y_pred_pos = torch.concatenate([y_pred_pos, zeros], axis=-1)
    neg_loss = torch.logsumexp(y_pred_neg, axis=-1)
    pos_loss = torch.logsumexp(y_pred_pos, axis=-1)
    return neg_loss + pos_loss


#****参考的源地址：https://gist.github.com/sam-writer/723baf81c501d9d24c6955f201d86bbb
#****以及 https://github.com/agemagician/ProtTrans/blob/master/Fine-Tuning/PT5_LoRA_Finetuning_per_prot.ipynb
class ClassConfig:
    def __init__(self, dropout=0.15, node_nums=1, layer_nums=1, batch_size=8):
        self.dropout_rate = dropout
        self.node_nums = node_nums
        self.layer_nums = layer_nums
        self.batch_size = batch_size

class SelfAttentionLayer(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(SelfAttentionLayer, self).__init__()
        self.query = nn.Linear(hidden_size, output_size)
        self.key = nn.Linear(hidden_size, output_size)
        self.value = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=-1)
        self.scale_factor = torch.sqrt(torch.tensor(output_size, dtype=torch.float32))

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        attn_weights = torch.matmul(Q, K.permute(0,1)) / self.scale_factor
        attn_weights = self.softmax(attn_weights)
        attn_output = torch.matmul(attn_weights, V)
        return attn_output

class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed



class HMCNClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, class_config):
        super().__init__()
        self.hierarchical_depth = [0, 64, 64, 128]
        self.global2local = [0, 64, 128, 128]
        self.hierarchical_class = [44, 42, 44]
        # 定义local层和global层
        self.local_layers = torch.nn.ModuleList()
        self.global_layers = torch.nn.ModuleList()

        for i in range(1, len(self.hierarchical_depth)):
            self.global_layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(config.hidden_size + self.hierarchical_depth[i-1], self.hierarchical_depth[i]),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm1d(self.hierarchical_depth[i]),
                    # RMSNorm(self.hierarchical_depth[i]),
                    torch.nn.Dropout(p=0.2)
                ))
            self.local_layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(self.hierarchical_depth[i], self.global2local[i]),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm1d(self.global2local[i]),
                    # RMSNorm(self.global2local[i]),
                    torch.nn.Linear(self.global2local[i], self.hierarchical_class[i-1])
                ))
        self.global_layers.apply(self._init_weight)
        self.local_layers.apply(self._init_weight)
        self.linear = torch.nn.Linear(self.hierarchical_depth[-1], class_config.layer_nums)
        self.linear.apply(self._init_weight)

    def _init_weight(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.1) 

    def forward(self, hidden_states):
        global_layer_activation = hidden_states
        local_layer_outputs = []
        for i, (local_layer, global_layer) in enumerate(zip(self.local_layers, self.global_layers)):
            local_layer_activation = global_layer(global_layer_activation)
            local_layer_outputs.append(local_layer(local_layer_activation))
            if i < len(self.global_layers)-1:
                global_layer_activation = torch.cat((local_layer_activation, hidden_states), 1)
            else:
                global_layer_activation = local_layer_activation

        global_layer_output = self.linear(global_layer_activation)
        local_layer_output = torch.cat(local_layer_outputs, 1)

        return global_layer_output, local_layer_output, 0.5 * global_layer_output + 0.5 * local_layer_output

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, class_config):
        super().__init__()
        # self.dense1 = nn.Linear(config.hidden_size, config.hidden_size)
        # self.batch_norm1 = nn.BatchNorm1d(config.hidden_size)
        # self.dropout = nn.Dropout(class_config.dropout_rate)
        # self.dense2 = nn.Linear(config.hidden_size, config.hidden_size // 2)
        # self.batch_norm2 = nn.BatchNorm1d(config.hidden_size // 2)
        # self.out_proj = nn.Linear(config.hidden_size, class_config.node_nums)
        self.dim = config.hidden_size // 2
        self.layer_score = nn.Linear(class_config.layer_nums, self.dim)
        # self.node_score =  nn.Linear(config.hidden_size, self.dim)
        # self.v = nn.Linear(config.hidden_size, self.dim)
        self.batch_norm1 = nn.BatchNorm1d(self.dim + config.hidden_size)
        self.dropout = nn.Dropout(class_config.dropout_rate)
        self.out_proj = nn.Linear(self.dim + config.hidden_size, class_config.node_nums)


    def forward(self, layer_logits, hidden_states):
        # hidden_states = self.dense1(hidden_states)
        # hidden_states = self.batch_norm1(hidden_states)
        # hidden_states = torch.tanh(hidden_states)
        # hidden_states = self.dense2(hidden_states)
        # hidden_states = self.batch_norm2(hidden_states)
        # hidden_states = torch.tanh(hidden_states)
        # hidden_states = self.dropout(hidden_states)
        # out = self.out_proj(hidden_states)
        # return out

        layer_score = self.layer_score(layer_logits)
        hidden_states = torch.concat((hidden_states, layer_score), dim=-1)
        hidden_states = self.batch_norm1(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = nn.ReLU()(hidden_states)
        hidden_states = self.out_proj(hidden_states)

        return hidden_states

@dataclass
class MySequenceClassifierOutput(ModelOutput):
    layer_loss: Optional[torch.FloatTensor] = None
    node_loss: Optional[torch.FloatTensor] = None
    node_logits: torch.FloatTensor = None
    layer_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class T5EncoderCLSModel(T5PreTrainedModel):

    # def __init__(self, config: T5Config):
    def __init__(self, config: T5Config, class_config, cluster_relations, hierar_relations, cluster_nodes_relations, alpha):
        super().__init__(config)
        self.config = config
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # self.encoder = BertModel(encoder_config, self.shared)
        # self.encoder = T5Stack(encoder_config, self.shared)
        self.encoder = T5Stack(encoder_config, self.shared)

        # self.dropout = nn.Dropout(0.15)
        self.hierar_relations = hierar_relations
        self.cluster_relations = cluster_relations
        self.cluster_nodes_relations = cluster_nodes_relations

        self.penalty = 1e-8

        self.layer_loss_fn = ClassificationLoss(label_size=class_config.layer_nums,loss_type=LossType.BCE_WITH_LOGITS, alpha=alpha)
        self.nodes_loss_fn = ClassificationLoss(label_size=class_config.layer_nums + class_config.node_nums,loss_type=LossType.SIGMOID_FOCAL_CROSS_ENTROPY, alpha=alpha)

        self.layer_classifier = HMCNClassificationHead(config, class_config)
        self.node_classifier = ClassificationHead(config, class_config)
        # Model parallel
        self.model_parallel = True
        self.device_map = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        nodes=None,
        layers=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = self.mean_pooling(outputs[0],attention_mask).to(torch.bfloat16)
        # hidden_states = torch.concat(
        #                     (self.mean_pooling(outputs[0],attention_mask), self.max_pooling(outputs[0],attention_mask)),
        #                     dim=1
        # ).to(torch.bfloat16)
        
        global_logits, local_logits, layer_logits = self.layer_classifier(hidden_states)
        
        layer_loss = self.layer_loss_fn(
                            global_logits,
                            layers,
                            True, # use_hierar
                            True, # is_multi
                            self.penalty, # 惩罚因子
                            global_logits,
                            self.cluster_relations
                            )
        layer_loss += self.layer_loss_fn(
                            local_logits,
                            layers,
                            True,
                            True,
                            self.penalty, # 惩罚因子
                            local_logits,
                            self.cluster_relations
                            )
        
        node_logits = self.node_classifier(layer_logits, hidden_states)
        mask_tensor = torch.zeros_like(node_logits)
        for idx, relation in enumerate(self.cluster_nodes_relations):
            relation_indices = torch.tensor(self.cluster_nodes_relations[relation])
            mask_tensor[:, relation_indices] += layer_logits[:, idx].unsqueeze(1)
        
        
        node_loss = 10 * self.nodes_loss_fn(
                            node_logits,
                            nodes,
                            True,
                            True,
                            self.penalty, # 惩罚因子
                            layer_logits,
                            self.cluster_nodes_relations,
                            self.hierar_relations
                            )
        
        mask_logits = torch.where(
                                mask_tensor < 0,
                                torch.full_like(mask_tensor, float('-1e7')),  # 小于0的位置设置为-1e-7
                                torch.zeros_like(mask_tensor)
                            )
        
        node_logits = node_logits + mask_logits

        return MySequenceClassifierOutput(layer_loss=layer_loss,
                                        node_loss=node_loss,
                                        node_logits=node_logits,
                                        layer_logits=layer_logits,
                                        hidden_states=outputs.hidden_states,
                                        attentions=outputs.attentions)
    
    def mean_pooling(self, hidden_states, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_embeddings / sum_mask
    
    def max_pooling(self, hidden_states, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        hidden_states[input_mask_expanded == 0] = -1e9
        max_embeddings, _ = torch.max(hidden_states, dim=1)
        return max_embeddings
    
    