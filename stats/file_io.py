# Input files into Python for analysis
#
# Reference: 
# http://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html
from scipy.io import loadmat

def read_paths():
    '''
    Read in available file paths from text file.
    Ask user to specify which file they would like
    to use.
    '''
    document = "/home/ty/code/belkin_comp/data/file_paths.txt"
    f = open(document, 'r')
    path_list = [list for item in f.readlines()]
    print "Here is a list of available files:\n"
    for item in range(len(path_list)):
        print 'Number\tFile'
        print item, "\t", path_list[item]
    get_selection = input('Which number file do you want to use? ')
    while get_selection > len(path_list) - 1:
       get_selection = input('That number does not correspond to an entry,'\
                             ' please enter one that does: ')
    return path_list[get_selection]

path = read_paths()

def matlab_in(path):
    '''
    Load a matlab file from disc
    '''
    return loadmat(path)
