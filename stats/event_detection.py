import ProcessMatlabFiles as pmf
import pandas as pd
import numpy as np


# Define Parameters
data = pmf.loadFile()["../flattened.mat"] 
key = 'L1_Imag'

def file_input(data,key):
    '''
    Input the data structure from ProcessMatlabFiles.py
    '''
    dic = pmf.ProcessRawData(data)[key]
    return dic

def rolling_mean(dic, key):
    '''
    Calculate the rolling mean
    '''
    t = pd.Series(dic,index=list(range(len(dic))))
    x = np.arange(0,t.shape[0])
    d = pd.Series(x, t)
    d_mva = pd.rolling_mean(d, 10)
    return d_mva

d_mva = rolling_mean(file_input(data,key), key)

print  "First 20 Rolling averages: \n ", d_mva[0:20]

# Labels
labels = ["\n","Mean", "Median","Std", "Var", "Min", "Max", "Skew", "Kurt", "Count"]

# Descriptive Statistics for L1_Imag, d_mva
des_t1 = ["d_mva", str(d_mva.mean())[0:5], d_mva.median(),
          str(d_mva.std())[0:5], str(d_mva.var())[0:5],
          d_mva.min(), d_mva.max(), str(d_mva.skew())[0:5],
          str(d_mva.kurt())[0:5], d_mva.count()]

for characteristic in range(len(labels)):
    print labels[characteristic], "\t", des_t1[characteristic]
        
        
