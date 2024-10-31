import pandas as pd
import json

path = '/data/csyData/uniprot_test/data/XML_testdata/go_temp/GO_data/cco/withlayer2/'
datatype = 'stride/'
train_data_dir1 = path + datatype + 'traindata.csv'

with open('/data/csyData/uniprot_test/code/GOcode/cco_version2/freq.json') as f:
    freq = json.load(f)

data = pd.read_csv(train_data_dir1)
high_freq = {}
low_freq = {}

for go,value in freq.items():
    if value < 0.5:
        high_freq[go] = value
    elif value > 0.94:
        low_freq[go] = value

data[data[list(low_freq.keys())].eq(1).any(axis=1)].index.tolist()
low_freq_label_indices = data[data[list(low_freq.keys())].eq(1).any(axis=1)].index.tolist()
temp = data.loc[low_freq_label_indices]
temp['sequence'] = temp['sequence'].apply(lambda x: x[::-1])
data = pd.concat([data] + [temp]).reset_index(drop=True)

print(temp)