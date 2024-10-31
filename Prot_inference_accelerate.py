import os
import time
import math
import torch
import pandas as pd
import numpy as np
from torch.optim import Adam
from accelerate import Accelerator
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split, Subset
from transformers import T5EncoderModel, T5Tokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, multilabel_confusion_matrix
from ProtTransModel2 import ClassConfig, T5EncoderClassificationHead, T5EncoderCLSModel, set_seeds
from accelerate.logging import get_logger
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
import json

logger = get_logger(__name__)
# os.environ["ACCELERATE_TORCH_DEVICE"] = "cuda:1"

num_labels = 5101   # 10 > 5091

epoch = 400    # 可能修改
max_length = 512
train_batch_size = 128
valid_batch_size = 128
# lr = 3e-4
lr = 2e-4

model_dirname = '/step_Focalloss2_'
model_path = '/data/csydata/prot-traning/prot-trans/prot_t5_xl_half_uniref50-enc'

with open('/data/csydata/GO_test/data/withclusterdata/label_freq_list2.json') as f:
    label_freq_list = json.load(f)

path = '/data/csydata/GO_test/data/withclusterdata/'

train_data_dir1 = path + 'traindata.csv'
train_data_dir2 = path + 'trainNolabel.csv'
train_data_dir3 = path + 'traindata2.csv'

test_data_dir1 = path + 'testdata.csv'
test_data_dir2 = path + 'testNolabel.csv'
test_data_dir3 = path + 'testdata2.csv'

load_model = False
saved_ep = 0   # 重新训练时，记得改成最新的模型参数


with open('/data/csyData/uniprot_test/data/XML_testdata/go_temp/GO_data/cco/GO_cco_withlayerDepthDict3.json') as f:
    temp_hierar_relations = json.load(f)

class MyDataset(Dataset):

    def __init__(self,data_dirlist) -> None:
        super().__init__()
        self.data = pd.concat(data_dirlist,axis=0)

    def __getitem__(self, index):
        return self.data["sequence"].iloc[index], self.data.iloc[index,2:].tolist()
    
    def __len__(self):
        return len(self.data)

def prepare_dataloader():
    train_list = []
    for i in [train_data_dir1,train_data_dir2,train_data_dir3]:
        train_list.append(pd.read_csv(i))
    trainset = MyDataset(train_list)
    valid_list = []
    for i in [test_data_dir1,test_data_dir2,test_data_dir3]:
        valid_list.append(pd.read_csv(i))
    validset = MyDataset(valid_list)

    tokenizer = T5Tokenizer.from_pretrained(model_path)

    def collate_func(batch):
        sequence, labels = [], []
        for item in batch:
            sequence.append(item[0])
            labels.append(item[1])
        inputs = tokenizer(sequence, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
        # del inputs["token_type_ids"]
        inputs["labels"] = torch.FloatTensor(labels)
        return inputs

    trainloader = DataLoader(trainset, batch_size=train_batch_size, collate_fn=collate_func, shuffle=True)
    validloader = DataLoader(validset, batch_size=valid_batch_size, collate_fn=collate_func, shuffle=False)

    return trainloader, validloader


def prepare_model_and_optimizer():

    model = T5EncoderModel.from_pretrained(model_path)
    model.config.d_model = 1024
    class_config = ClassConfig(num_labels=num_labels, batch_size=train_batch_size)
    class_model = T5EncoderCLSModel(model.config, class_config)
    class_model.encoder = model.encoder

    model = class_model
    del class_model
    model = model.cuda()

    # 冻结模型参数
    for name, param in model.named_parameters():
        if 'encoder' in name or 'model' in name:
            param.requires_grad = False
    
    # 打印模型参数
    lst = []    
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            lst.append(param.nelement())
    print(f"trainable paras number: {sum(lst)}")

    optimizer = Adam(model.parameters(), lr=lr)

    # 读取模型数据2
    if load_model:
        logger.info(f"Loading trained model :./ckpts" + model_dirname + f"{saved_ep}/")
        ckpt = torch.load('./ckpts' + model_dirname + f"{saved_ep}/model.pt", weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        # optimizer.load_state_dict(ckpt['optim_state'])

    return model, optimizer


def compute_pred(output):

    return output.logits > 0

def Accuracy(y_true, y_pred):
    count = 0
    for i in range(y_true.size(0)):
        p = sum(torch.logical_and(y_true[i], y_pred[i]))
        q = sum(torch.logical_or(y_true[i], y_pred[i]))
        count += p / q
    return count

def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)

    return roc_auc


def evaluate(model, validloader, accelerator: Accelerator):
    model.eval()
    subset_acc_num_layer0 = 0
    # subset_acc_num_layer1 = 0

    tp_sum_layer0 = 0
    pred_sum_layer0 = 0
    true_sum_layer0 = 0
    validlabels = []
    predlabels = []
    tp_sum_layer1 = [0] * 4
    pred_sum_layer1 = [0] * 4
    true_sum_layer1 = [0] * 4

    with torch.inference_mode():
        for batch in validloader:
            output = model(**batch)

            # 打印
            # accelerator.print(output.logits.mean())
            pred0 = output.logits > 0
            pred0, refs0 = accelerator.gather_for_metrics((pred0, batch["labels"]))
            pred0 = pred0.cpu()
            refs0 = refs0.cpu()
            validlabels.append(torch.sigmoid(output.logits).cpu().numpy())
            predlabels.append(refs0.numpy())

            # pred.append(torch.softmax(output.logits))

            # 计算样本的 subset acc
            subset_acc_num_layer0 += accuracy_score(pred0,refs0,normalize=False)
            
            # 计算一个batch里面的precision和recall
            MCM0 = multilabel_confusion_matrix(pred0.int(), refs0.int())
            MCM1 = MCM0[10:]
            tp_sum_layer0 += MCM0[:, 1, 1]
            pred_sum_layer0 += MCM0[:, 1, 1] + MCM0[:, 0, 1]
            true_sum_layer0 += MCM0[:, 1, 1] + MCM0[:, 1, 0]
            for idx, freq in enumerate(label_freq_list):
                tp_sum_layer1[idx] += MCM1[freq, 1, 1]
                pred_sum_layer1[idx] += MCM1[freq, 1, 1] + MCM1[freq, 0, 1]
                true_sum_layer1[idx] += MCM1[freq, 1, 1] + MCM1[freq, 1, 0]

        roc_auc = compute_roc(np.concatenate(validlabels,axis=0), np.concatenate(predlabels,axis=0))
        for i in range(4):
            print('第'+str(i+1)+'层的验证集 TP / TP + FP:', [num / denom if denom != 0 else 0 for num, denom in zip(tp_sum_layer1[i], pred_sum_layer1[i])])
            print('第'+str(i+1)+'层的验证集 TP / TP + FP:', [num / denom if denom != 0 else 0 for num, denom in zip(tp_sum_layer1[i], true_sum_layer1[i])])
        
        print(roc_auc)

        # accelerator.print('验证集 TP / TP + FP:', tp_sum_layer1[0], ' / ', pred_sum_layer1[0])
        # accelerator.print('验证集 TP / TP + FN:', tp_sum_layer1[0], ' / ', true_sum_layer1[0])

    return (subset_acc_num_layer0 / len(validloader.dataset)
    )


def train(model, optimizer, trainloader, validloader, accelerator: Accelerator, epoch=epoch, log_step=10):

    global_step = 0
    start_time = time.time()

    # b = 0.001
    for ep in range(epoch):
        model.train()
        ep = ep + saved_ep
        subset_acc_num_layer0 = 0

        for batch in trainloader:
            with accelerator.accumulate(model):
                output = model(**batch)
                loss = output.loss
                optimizer.zero_grad()
                # loss = (loss - b).abs() + b
                accelerator.backward(loss)
                optimizer.step()

                pred0 = compute_pred(output)
                pred0, refs0 = accelerator.gather_for_metrics((pred0, batch["labels"]))
                pred0 = pred0.cpu()
                refs0 = refs0.cpu()
                # pred1 = pred0[:,0:10]
                # refs1 = refs0[:,0:10]

                # 计算样本的 subset acc
                subset_acc_num_layer0 += accuracy_score(pred0,refs0,normalize=False)
                if accelerator.sync_gradients:
                    global_step += 1

                    if global_step % log_step == 0:
                        loss = accelerator.reduce(loss, "mean")
                        accelerator.print(f"ep: {ep}, global_step: {global_step}, loss: {loss.item()}")
                        accelerator.log({"loss": loss.item()}, global_step)
                        # accelerator.log({"loss": flood.item()}, global_step)

                    if global_step % 600 == 0 and global_step != 0:
                        accelerator.print(f"save checkpoint -> step_{ep}")

                        if accelerator.is_local_main_process:
                            
                            unwrap_model = accelerator.unwrap_model(model)      
                            unwrap_optim = optimizer

                            if not os.path.exists('./ckpts' + model_dirname + f"{ep}/"):
                                os.makedirs('./ckpts' + model_dirname + f"{ep}/")
                            
                            torch.save({        
                                'model_state' : unwrap_model.state_dict(),        
                                'optim_state' : unwrap_optim.state_dict()}, 
                                './ckpts' + model_dirname + f"{ep}/model.pt")	
                            logger.info(f'step_{saved_ep}.savetensors is saved...')
                            accelerator.print(f"step_{ep} saved!")
        
        valid_acc0 = evaluate(model, validloader, accelerator)
        accelerator.print(f"ep: {ep}, time: {time.time() - start_time}")

        accelerator.print('训练集的acc:', subset_acc_num_layer0 / len(trainloader.dataset))
        accelerator.log({"训练集的acc": subset_acc_num_layer0 / len(trainloader.dataset)}, global_step)
        # accelerator.print('训练集第一层的acc:',subset_acc_num_layer1 / len(trainloader.dataset))
        # accelerator.log({"训练集第一层的acc": subset_acc_num_layer1 / len(trainloader.dataset)}, global_step)

        accelerator.print('验证集的acc:', valid_acc0)
        accelerator.log({"验证集的acc": valid_acc0}, global_step)
        # accelerator.print('valid auc:', roc_auc)

        # for i in p:
        #     sp = [0 if item == 'nan' else float(item) for item in i]
        #     accelerator.log({"ave precision": sum(sp)/len(sp)}, global_step)
        #     accelerator.log({"max precision": max(sp)}, global_step)
        # for i in r:
        #     sr = [0 if item == 'nan' else float(item) for item in i]
        #     accelerator.log({"ave recall": sum(sr)/len(sr)}, global_step)
        #     accelerator.log({"max recall": max(sr)}, global_step)

    accelerator.end_training()

def main():

    accelerator = Accelerator(gradient_accumulation_steps=2, log_with="tensorboard", project_dir="ckpts")

    accelerator.init_trackers("runs")

    trainloader, validloader = prepare_dataloader()

    model, optimizer = prepare_model_and_optimizer()

    model, optimizer, trainloader, validloader = accelerator.prepare(model, optimizer, trainloader, validloader)

    # train(model, optimizer, trainloader, validloader, accelerator)

    evaluate(model, validloader, accelerator)

    # train(model, optimizer, trainloader, validloader, accelerator, resume="/data/csyData/uniprot_test/code/the_third_version/ckpts/step1_100/model")

if __name__ == "__main__":
    main()