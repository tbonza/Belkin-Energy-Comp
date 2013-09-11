% Simple Matlab script to flatten the files provided by kaggle
inFile = '/home/ty/code/belkin_comp/data/H3/Tagged_Training_07_30_1343631601.mat'
outFile = '../flattened.mat'
mat = load(inFile);
HF = mat.Buffer.HF;
TimeTicksHF = mat.Buffer.TimeTicksHF;
LF1V = mat.Buffer.LF1V;
LF1I = mat.Buffer.LF1I;
TimeTicks1 = mat.Buffer.TimeTicks1;
LF2V = mat.Buffer.LF2V;
LF2I = mat.Buffer.LF2I;
TimeTicks2 = mat.Buffer.TimeTicks2;
TaggingInfo = mat.Buffer.TaggingInfo;
save(outFile, 'HF', 'TimeTicksHF', 'LF1V', 'LF1I', 'TimeTicks1', 'LF2V', 'LF2I', 'TimeTicks2', 'TaggingInfo','-v7');


