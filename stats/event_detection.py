import ProcessMatlabFiles as mark
import pandas as pd
import numpy as np
import pylab
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# Define Parameters
data = mark.loadFile()["../flattened.mat"] 
key = 'L1_Imag'

def file_input(data,key):
    '''
    Input the data structure from ProcessMatlabFiles.py
    '''
    dic = mark.ProcessRawData(data)[key]
    return dic

def rolling_mean(dic, key):
    '''
    Calculate the rolling mean
    '''
    t = pd.Series(dic,index=list(range(len(dic))))
    x = np.arange(0,t.shape[0])
    d = pd.Series(t, x)
    d_mva = pd.rolling_mean(d, 10)
    return d_mva

def print_nice(data,key):
    '''
    Taking all of the print statements and putting them
    in a function so this stops hurting my eyes.
    '''
    d_mva = rolling_mean(file_input(data,key), key)

    print  "First 20 Rolling averages: \n ", d_mva[0:20]

    # Labels
    labels = ["\n","Mean", "Median","Std", "Var", "Min", "Max", "Skew",
              "Kurt", "Count"]

    # Descriptive Statistics for L1_Imag, d_mva
    des_t1 = ["d_mva", str(d_mva.mean())[0:5], d_mva.median(),
              str(d_mva.std())[0:5], str(d_mva.var())[0:5],
              d_mva.min(), d_mva.max(), str(d_mva.skew())[0:5],
              str(d_mva.kurt())[0:5], d_mva.count()]
    
    for characteristic in range(len(labels)):
        print labels[characteristic], "\t", des_t1[characteristic]
        
def plot_nice(data, key):
    '''
    Make a histogram so we can get a better look at the data
    '''
    d_mva = rolling_mean(file_input(data,key), key)

    # Parameters used to plot the normal probability density function
    mu1 = d_mva.mean() # mean of distribution
    sigma1 = d_mva.std() # standard deviation of distribution
    # Number of bins used in the histogram
    num_bins = 20
    # the histogram of the data 
    n1, bins1, patches1 = plt.hist(data['age_1'], num_bins, normed=1,
                                   facecolor='red', alpha=0.5,
                                   label='age_t1')
    # add a 'best fit' line for age_t1
    y1 = mlab.normpdf(bins1, mu1, sigma1)
    plt.plot(bins1, y1, 'r--')
    # Add labels to the figure
    plt.xlabel('Age')
    plt.ylabel('Probability')
    plt.title(r'Figure 1: Histogram of age_t1, age_t2')
    plt.legend(loc='upper left')
    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.show()

print_nice(data,key)
plot_nice(data,key)
