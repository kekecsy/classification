import pandas as pd
import json
import numpy as np
import random
import pickle
import lancedb
import pyarrow as pa

path = '/data/csyData/uniprot_test/data/XML_testdata/go_temp/GO_data/cco/withlayer2/'
datatype = 'stride/'
train_data_dir1 = path + datatype + 'traindata.csv'
train_data_dir2 = path + datatype + 'trainNolabel.csv'
train_filelist = [train_data_dir1, train_data_dir2]


with open('/data/csyData/uniprot_test/code/GOcode/cco_version2/freq.json') as f:
    freq = json.load(f)

high_freq = {}
low_freq = {}

for go,value in freq.items():
    if value < 0.5:
        high_freq[go] = value
    elif value > 0.94:
        low_freq[go] = value

traindata_list = []
for i in train_filelist:
    traindata_list.append(pd.read_csv(i))

data = pd.concat(traindata_list,axis=0).reset_index(drop=True)
column = list(data.columns)
label_cols = column[2:]


def longest_common_continuous_subsequence(A, B):
    m, n = len(A), len(B)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_len = 0  # 记录最长公共连续子序列的长度
    end_pos = 0  # 记录最长公共连续子序列在 A 中的结束位置

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if A[i - 1] == B[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                # 更新最长子序列长度和结束位置
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    end_pos = i  # 更新结束位置
            else:
                dp[i][j] = 0  # 不连续，重置为 0

    # 提取最长公共连续子序列
    longest_subseq = A[end_pos - max_len:end_pos] if max_len > 0 else ""
    return max_len, longest_subseq

#  temp_cluster_nodes是聚类的节点字典，cluster_nodes是排序后的聚类key字典
with open('/data/csyData/pygosemsim-master/pygosemsim-master/cco_Kmeans_cluster2.json') as f:
    temp_cluster_nodes = json.load(f)

main_numbers, sub_numbers = [], []
for label in list(temp_cluster_nodes.keys()):
    if '_' not in label:
        main_numbers.append(label)
    else:
        sub_numbers.append(label)

main_numbers.sort(key=lambda x: int(x))
sub_numbers.sort(key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1])) if '_' in x else (int(x), -1))
cluster_nodes = main_numbers + sub_numbers


# go2index函数
def go2index(target_go):
    go = label_cols
    if isinstance(target_go, list):
        return [go.index(i) for i in target_go]
    elif isinstance(target_go, str):
        if target_go in go:
            return go.index(target_go)
        else:
            return None

temp_nodelabel = np.array(data.iloc[:,2:])
temp_layerlabel = []
for cluster_name in cluster_nodes:
    golist = temp_cluster_nodes[cluster_name]
    temp = (np.sum(temp_nodelabel[:,np.array(go2index(golist))],axis=1) > 0).astype(int)
    temp_layerlabel.append(temp)
    # print(f"{cluster_name} nums:", temp.sum())

temp_layerlabel = pd.DataFrame(temp_layerlabel).T
temp_layerlabel.columns=cluster_nodes
# print(temp_layerlabel)

def labels_to_int_main(row):
    return ''.join(map(str, row[:len(main_numbers)]))

def labels_to_int_sub(row):
    return ''.join(map(str, row[len(main_numbers):len(cluster_nodes)]))

def hamming_distance(x, y):
    return bin(x ^ y).count('1')

# 创建连接lancedb
uri = "./lancedb"
db = lancedb.connect(uri)
main_schema = pa.schema([pa.field("index", pa.int64()),
                    pa.field("vector", pa.list_(pa.float16(), list_size=len(main_numbers))),
                    pa.field("main_frequence", pa.int64()),
                    ])
sub_schema = pa.schema([pa.field("index", pa.int64()),
                    pa.field("vector", pa.list_(pa.float16(), list_size=len(sub_numbers))),
                    pa.field("sub_frequence", pa.int64())
                    ])

tbl1 = db.create_table("main_vector", schema=main_schema, exist_ok=True)
tbl2 = db.create_table("sub_vector", schema=sub_schema, exist_ok=True)

def find_similar_samples(df, threshold=2):
    # 将标签组合转换为整数
    df['labels_to_int_main'] = df.apply(labels_to_int_main, axis=1)
    df['labels_to_int_sub'] = df.apply(labels_to_int_sub, axis=1)

    # 创建哈希表存储每个整数对应的行索引
    main_hash_table = {}
    for idx, label_int in enumerate(df['labels_to_int_main']):
        if label_int not in main_hash_table:
            main_hash_table[label_int] = []
        main_hash_table[label_int].append(idx)
    
    sub_hash_table = {}
    for idx, label_int in enumerate(df['labels_to_int_sub']):
        if label_int not in sub_hash_table:
            sub_hash_table[label_int] = []
        sub_hash_table[label_int].append(idx)

    temp_df=pd.DataFrame()
    df['index'] = range(len(df))
    temp_df['index'] = range(len(df))
    temp_df['vector'] = df.apply(lambda row: np.array([row[col] for col in main_numbers], dtype=np.float16), axis=1)
    temp_df['main_frequence'] = df.apply(lambda row: len(main_hash_table[''.join(map(str, row[:len(main_numbers)]))]), axis=1)
    tbl1.add(temp_df[['index','vector','main_frequence']])

    temp_df['vector'] = df.apply(lambda row: np.array([row[col] for col in sub_numbers], dtype=np.float16), axis=1)
    temp_df['sub_frequence'] = df.apply(lambda row: len(sub_hash_table[''.join(map(str, row[len(main_numbers):len(cluster_nodes)]))]), axis=1)
    tbl2.add(temp_df[['index','vector','sub_frequence']])

    same_samples = []
    different_samples = []
    
    for idx in range(len((df['labels_to_int_sub']))):
        
        main_label_int = df.iloc[idx,:][:len(main_numbers)].astype('float16').tolist()
        sub_label_int = df.iloc[idx,:][len(main_numbers):len(cluster_nodes)].astype('float16').tolist()
        if len(sub_hash_table[sub_label_int]) > 1:
            for other_idx in sub_hash_table[sub_label_int]:
                if idx != other_idx:
                    same_samples.append((other_idx))
        elif len(main_hash_table[main_label_int]) > 1:
            for other_label_int in sub_label_int:
                if hamming_distance(sub_label_int, other_label_int) < threshold:
                    same_samples.append((other_idx))
        else:
            print('hello')
        for other_label_int in main_hash_table:
            if other_label_int != main_label_int and hamming_distance(main_label_int, other_label_int) > threshold:
                for other_idx in main_hash_table[other_label_int]:
                    if idx != other_idx:
                        different_samples.append((idx, other_idx))
    
    return same_samples, different_samples

same_samples, different_samples = find_similar_samples(temp_layerlabel)

# for col in low_freq_dup_set:
#     temp = low_freq_dup[low_freq_dup['index'] == col]['sequence']
#     go = low_freq_dup[low_freq_dup['index'] == col]['go']
#     for i in range(len(temp) - 1):
#         new_row = {'source':temp.iloc[i], 'augment':temp.iloc[i+1], 'go': tuple(go.iloc[i])}
#         compare_data.loc[len(compare_data)] = new_row

# data2 = data2.groupby('go').head(5).reset_index(drop=True)

# output = pd.DataFrame({'source':[],'augment':[]})
# for i, row1 in compare_data.iterrows():
#     temp_index = []
#     for j, row2 in data2.iterrows():
#         if set(row1['go']).isdisjoint(set(row2['go'])):  # 检查两个集合是否有交集
#             temp_index.append(j)
#         new_row = {'source':row1['source'], 
#                    'augment':row1['augment']
#                    }
#         output.loc[len(output)] = new_row

# output.to_csv(path + 'augment/augment_data_temp.csv',index=False)


# output.to_csv(path + 'augment/augment_data.csv',index=False)

# a = ''.join(data[data['index'] == 24004]['sequence'].iloc[-1].split())
# b = ''.join(data[data['index'] == 24004]['sequence'].iloc[-2].split())
# print(longest_common_continuous_subsequence(a,b))

# print(a == b)

# data = pd.concat([data] + [temp]).reset_index(drop=True)

# print(temp)