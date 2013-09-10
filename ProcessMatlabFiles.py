import scipy.io as sio
import numpy as np
import pylab as p
import math
import os
import matplotlib as mp

def loadFile():
    """ This function loads the flattened Matlab files defined by the line files
        into a large dictionary and returns them.
        Output: A dictionary with the variables contained in the Matlab file 
                and the Python gneerated variables"""

    DATADIR = "../H3"
    PWD = os.getcwd()
    bigList= {}
    try: 
#    os.chdir(DATADIR)
#    for files in os.listdir(os.getcwd()):
#        if(files[-4:] == '.mat'): 
            files = '../flattened.mat' # DATADIR + "/Tagged_Training_07_30_1343631601.mat"
            mat_contents = sio.loadmat(files)
            head = mat_contents['__header__']
            version = mat_contents['__version__']
            glob = mat_contents['__globals__']
            bigList[files] = {'header': head, 'version': version, 'globals': glob}
            for key in mat_contents.keys():
                if (key[0:2] != '__'):
                    bigList[files][key] = mat_contents[key]
                    
                  
            os.chdir(PWD)
            return bigList

  
 
    finally:
        os.chdir(PWD)

def plotAllRaw(dic):
    """This function plots the raw variables (current and voltage) in dic.
    It will plot fields that have more than 1 y variable but less than 20,"""
    count = 1;
    for key in dic.keys():
#        print key
#        figure(count)
        arr =  dic[key]
        if(type(arr) == type(np.eye(1))):
            numX =arr.shape[1]
  #          print numX
            ml = p.MultipleLocator(100000)
                            
            if (numX <20 and numX > 1):
                if (key == 'TaggingInfo'):
                    break
                fig = p.figure()
                fig.suptitle(key)
                nrows = math.ceil(numX / 3)
                ncols = numX if nrows == 1 else 3
 #               print nrows, ncols
                for i in range(numX):
                    sp = p.subplot(nrows, ncols, i+1)
                    pl = p.plot(abs(arr[:,i]))[0]
                    if(sp.is_last_row()):
                        p.xlabel("Time in milliseconds")
                    if(sp.is_first_col()):  
                        p.ylabel(key)
                    pl.axes.xaxis.set_major_locator(ml)
                    

                count = count +1

def plotHF(dic):
    """This function plots the HF (High Frequency) field in the dictionary
    data structure. Note that it may use up all your memory if two large a function is pass (on my machine with 3GB of RAM 2000x500 ndarray is the limit)."""
    fig = p.figure()
    fig.suptitle("HF")
    p.pcolor(dic["HF"], cmap="jet")
    p.xlabel("Time in milliseconds")
    p.ylabel("Frequency in kHz")

def ProcessRawData(dic):
    """This function calculates the derived variables Real, Reactive, and
    Apparent Power along with the Power Factor, and returns a dict containing
    them."""
    # 
    L1_P = dic["LF1V"] * dic["LF1I"].conj()
    L2_P = dic["LF2V"] * dic["LF2I"].conj()

    # Compute net Complex power
    L1_ComplexPower = L1_P.sum(axis=1)
    L2_ComplexPower = L2_P.sum(axis=1)

    # Calculate Real, Reactive and Apparent Powers
    L1_Real = L1_ComplexPower.real
    L1_Imag = L1_ComplexPower.imag
    L1_App = abs(L1_ComplexPower)

    L2_Real = L2_ComplexPower.real
    L2_Imag = L2_ComplexPower.imag
    L2_App = abs(L2_ComplexPower)
    
    # Calculate Power Factor, on the first 60Hz component
    L1_Pf = np.cos(np.angle(L1_P[:,0]))
    L2_Pf = np.cos(np.angle(L2_P[:,0]))

    #Create data structure
    ProcessedData = {"L1_Real": L1_Real, "L1_Imag": L1_Imag, "L1_App": L1_App,
                     "L2_Real": L2_Real, "L2_Imag": L2_Imag, "L2_app": L2_App,
                     "L1_Pf" : L1_Pf, "L2_Pf": L2_Pf, "L1_Time": dic["TimeTicks1"],
                     "L2_Time" : dic["TimeTicks2"]}
    if("TaggingInfo" in dic):
        ProcessedData["TaggingInfo"] = dic["TaggingInfo"]

    return ProcessedDatai

def ExamplePlots(dic):
    """This function makes the first 3 example plots (the fourth is just he HF
    Noise plot which you can create by calling plotHF). This requies as input 
    the dict created by the processRawData function."""
    hasLabels = False
    if("TagginInfo" in dic):
        hasLabels = True
    fig = p.figure()
    s1 = p.subplot(311)
    p.plot(dic["L1_Time"], dic["L1_Real"], "b", dic["L2_Time"], dic["L2_Real"], "r")
    p.title("Real Power(W) and ON/OFF Device Category IDs")
    if(hasLabels):
        #TODO plot device cateogry lables
        True 
    s2 = p.subplot(312)
    p.plot(dic["L1_Time"], dic["L1_Imag"], "b", dic["L2_Time"], dic["L2_Real"], "r")
    p.title("Imaginary/Reactive Power (VAR)")

    s3 = p.subplot(313)
    p.title("Power Factor")
    p.plot(dic["L1_Time"], dic["L1_Pf"], 'b', dic["L2_Time"], dic["L2_Pf"], 'r')
    p.xlabel("Unix Timestamp")

# Code to run and create plots

# create dicts containing the files
dict2 = loadFile()
fN = dict2.keys()[0]
bigDic = dict2[fN]
del dict2, fN
ProcessedData = ProcessRawData(bigDic) 
#reducedHF = bigDic["HF"][1:2000,1:500]  # a smaller version of HF to pass to plotHF
# del bigDic # To delete to free up memory if no longer needed
#rDic = {"HF": reducedHF}

# Example calls to each of the plot functions
#plotAllRaw(bigDic) #uncomment to plot the raw variables
#plotHF(rDic) # uncommet to make and HF Noise plot
#ExamplePlots(ProcessedData) 
p.show()
