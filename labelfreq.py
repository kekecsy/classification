import json
import pandas as pd
import numpy as np
with open('/data/csydata/GO_test/data/withlayerdata/mfo/sta_count3.json') as f:
    sta_count = json.load(f)

ssrlist = []
srlist = []
rlist = []
nlist = []
elselist = []
for i in sta_count:
    if sta_count[i]['count1'] < 1e2:
        ssrlist.append(i)
    elif sta_count[i]['count1'] < 1e3:
        srlist.append(i)
    elif sta_count[i]['count1'] < 2e4:
        rlist.append(i)
    elif sta_count[i]['count1'] < 5e5:
        nlist.append(i)
    else:
        elselist.append(i)   # 空

data = pd.read_csv('/data/csydata/GO_test/data/withclusterdata/testNolabel.csv')

column = list(data.columns)
column.remove('index')
column.remove('sequence')
for i in range(1,11):
    column.remove(str(i))

li = []
le = 0
for sublist in [ssrlist,srlist,rlist,nlist]:
    indices = [column.index(item) for item in sublist]
    li.append(indices)
    le += len(indices)


with open("/data/csydata/GO_test/first_version/label_freq_list2.json", "w+") as jsonFile:
    jsonFile.write(json.dumps(li, indent = 4))