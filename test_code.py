import numpy as np
from numpy import load

data = load('feats.npz')
lst = data.files
for item in lst:
    #print(item)
    #print(data[item])
    d1 = data[item]

data2 = load('sample.npz')
lst = data2.files
for item in lst:

    d2 = data[item]
print("**********************************************")
print(d1)
print("**********************************************")
print(d2)
print("**********************************************")
print(d1-d2)