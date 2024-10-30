import pandas as pd

source_path = '/data/csyData/uniprot_test/data/XML_testdata/go_temp/GO_data/cco/withlayer2/stride/'
file_list = ['testdata.csv','trainNolabel.csv','traindata.csv','testNolabel.csv']
# output_path = '/data/csyData/uniprot_test/data/XML_testdata/go_temp/GO_data/cco/withlayer2/'


# del_col = []
# for i in range(1,9):
#     del_col.append(str(i))

# for file in file_list:
#     data = pd.read_csv(source_path + file)
#     for d in del_col:
#         del data[d]
#     data_col = list(data.columns)
#     data_col.remove('index')
#     data_col.remove('sequence')
#     data[data_col] = data[data_col].astype('int')
#     data.to_csv(output_path+file,index=False)


# 查看数据

data = pd.read_csv(source_path + file_list[3])
print(list(data.columns).index('GO:0044162'))