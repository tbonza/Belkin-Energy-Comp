import ProcessMatlabFiles as mark
import pandas as pd
import numpy as np
import pylab
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import pylab as pl
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

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
    n1, bins1, patches1 = plt.hist(d_mva, num_bins, normed=1,
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

#print_nice(data,key)
#plot_nice(data,key)

def knn_approach(data, key):
    '''
    Here we want to classify the data using discrete labels: on, off.
    Classification is computed from a simple majority vote of the
    nearest neighbors of each point: a query point is assigned the data
    class which has the most representatives within the nearest neighbors
    of the point.
    '''
    n_neighbors = 2

    # import some data to play with
    data = file_input(data,key)

    X = data[:259461]
    y = data[259461:]

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    for weights in ['uniform', 'distance']:
        # we create an instance of Neighbours Classifier and fit the data.
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
        clf.fit(X, y)

        # Plot the decision boundary. For that, we will assign a color
        # to each point in the mesh [x_min, m_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        pl.figure()
        pl.pcolormesh(xx, yy, Z, cmap=cmap_light)
        
        # Plot also the training points
        pl.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
        pl.xlim(xx.min(), xx.max())
        pl.ylim(yy.min(), yy.max())
        pl.title("3-Class classification (k = %i, weights = '%s')"
                 % (n_neighbors, weights))
        
        pl.show()
    

knn_approach(data, key)    


# It's not working. Here's the links you had open:
http://scikit-learn.org/stable/supervised_learning.html#supervised-learning
http://scikit-learn.org/stable/modules/neighbors.html
http://scikit-learn.org/stable/modules/neighbors.html#classification
http://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#example-neighbors-plot-classification-py
https://www.google.com/#q=ipython+emacs+24
http://stackoverflow.com/questions/13422653/ipython-support-on-emacs-24-x
    
