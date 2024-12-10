import lancedb
import pyarrow as pa
import numpy as np

uri = "./lancedb"
db = lancedb.connect(uri)

table_names = ['main_vector']
table = db.open_table(table_names[0])
data = np.vstack(table.to_pandas()[:64]['vector'].values)

freq = []
for d in data:
    freq.append(table.search(d).limit(1).to_list()[0]['main_frequence'])
print(freq)