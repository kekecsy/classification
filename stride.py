import pandas as pd
import copy
import numpy as np
def sliding_window_tokenizer(data, max_len, stride):

    result = {
        'index':[],
        'sequence':[]
    }
    for col in data.columns:
        if (col != 'index') & (col != 'sequence'):
            result[col] = []

    for index, sub_data in data.iterrows():
        sequence = "".join(sub_data['sequence'].split())
        labels = sub_data.drop(['index','sequence']).to_dict()
        for i in range(0, len(sequence), stride):
            result['index'].append(sub_data['index'])
            window_seq = sequence[i:i + max_len]
            if len(window_seq) < max_len:
                window_seq = sequence[-max_len:]
                result['sequence'].append(" ".join(window_seq))
                for label, value in labels.items():
                    result[label].append(value)
                break
            else:
                result['sequence'].append(" ".join(window_seq))
                for label, value in labels.items():
                    result[label].append(value)
    result_df = pd.DataFrame(result)
    return result_df


max_len = 512
stride = 128

path = '/data/csyData/uniprot_test/data/XML_testdata/go_temp/GO_data/cco/withcluster/'

datalist = ['traindata.csv','trainNolabel.csv','testdata.csv','testNolabel.csv']

for i in datalist:
    data = pd.read_csv(path + i)
    output = sliding_window_tokenizer(data, max_len, stride)
    output.to_csv(path + 'stride/' + i, index=False)