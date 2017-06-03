# coding: utf-8

import csv
import numpy as np

csvfile = file('iris.data', 'rb')
reader = csv.reader(csvfile)

datas = []
for line in reader:
    datas.append(np.array(line[0:2], dtype=np.float))

maritx = np.array(datas);
print(maritx)

csvfile.close()