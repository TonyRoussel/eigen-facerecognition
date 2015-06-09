from trainer import train
import submission as subm
import numpy as np
import files_load as fl
import sys
from random import shuffle

if len(sys.argv) != 3:
    exit(1)
trainD = fl.loadmatrixs(sys.argv[1])
validD = fl.loadmatrixs(sys.argv[2])
ftrainD, mtrainD = zip(*trainD)

mean, eigenvec, thetas = train(mtrainD)
print "Mean: ", mean
print "eigenvec: \n", eigenvec
count = 0
for idx, data in enumerate(validD):
    success = False
    filename, mtx = data
    thetaSubmit = subm.submit(mtx, mean, eigenvec)
    print "w:", thetaSubmit 
    matchIdx = subm.compareAvgGap(thetaSubmit, thetas)
    if filename[:filename.rfind("_")] == ftrainD[matchIdx][:ftrainD[matchIdx].rfind("_")]:
        success = True
        count = count + 1
    if success is True:
        print filename, " --> ", ftrainD[matchIdx], "[X]"
    else:
        print filename, " --> ", ftrainD[matchIdx], "[ ]"
print count, " / ", idx + 1, "===>", count / (idx + 1.) * 100, "%"
