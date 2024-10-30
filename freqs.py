from sklearn.cluster import KMeans
import json
import pandas as pd
import numpy as np


with open('/data/csyData/uniprot_test/data/XML_testdata/go_temp/GO_data/cco/sta_count3.json') as f:
    sta_count3 = json.load(f)

path = '/data/csyData/uniprot_test/data/XML_testdata/go_temp/GO_data/cco/withlayer2/'
datatype = 'stride/'
test_data_dir2 = path + datatype + 'testNolabel.csv'


def go2index(target_go):
    go = pd.read_csv(test_data_dir2, nrows=0).columns.tolist()[2:]
    if isinstance(target_go, list):
        return [go.index(i) for i in target_go]
    elif isinstance(target_go, str):
        if target_go in go:
            return go.index(target_go)
        else:
            return None

alpha = {}
for go,value in sta_count3.items():
    alpha[go] = value['count1']

total_samples = sum(alpha.values())

sorted_labels = sorted(alpha.items(), key=lambda x: x[1], reverse=True)

cumulative_frequencies = np.cumsum([freq for _, freq in sorted_labels]) / total_samples

threshold_1 = np.argmax(cumulative_frequencies >= 0.50)
threshold_2 = np.argmax(cumulative_frequencies >= 0.70)
threshold_3 = np.argmax(cumulative_frequencies >= 0.80)
threshold_4 = np.argmax(cumulative_frequencies >= 0.90)
threshold_5 = np.argmax(cumulative_frequencies >= 1.00)

# 划分头部和长尾
threshold_1_labels = {label: freq for label, freq in sorted_labels[:threshold_1 + 1]}
threshold_2_labels = {label: freq for label, freq in sorted_labels[threshold_1 + 1: threshold_2 + 1]}
threshold_3_labels = {label: freq for label, freq in sorted_labels[threshold_2 + 1: threshold_3 + 1]}
threshold_4_labels = {label: freq for label, freq in sorted_labels[threshold_3 + 1: threshold_4 + 1]}
threshold_5_labels = {label: freq for label, freq in sorted_labels[threshold_4 + 1:]}

labels_names = ['threshold_1_labels','threshold_2_labels','threshold_3_labels','threshold_4_labels','threshold_5_labels']

output = {}
for name,data,threshold in zip(labels_names,
                     [threshold_1_labels,threshold_2_labels,threshold_3_labels,threshold_4_labels,threshold_5_labels],
                     [[0.2,0.3],[0.3,0.5],[0.5,0.7],[0.7,0.9],[0.9,0.95]]):
    values = data.values()
    max_val = max(values)
    min_val = min(values)
    a = (threshold[0] - threshold[1]) / (max_val - min_val)
    b = threshold[1] - a * min_val
    for key in data:
        output[key] = round(a * data[key] + b, 4 )

with open("freq.json", "w+") as jsonFile:
    jsonFile.write(json.dumps(output, indent = 4))