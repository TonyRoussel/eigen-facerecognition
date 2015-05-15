from trainer import train
import submission as subm
import numpy as np
import files_load as fl
import sys
from random import shuffle

if len(sys.argv) == 1:
    exit(1)
fm = fl.loadmatrixs(sys.argv[1])
shuffle(fm)
trainD, validD = fl.split_list(fm, 90)
ftrainD, mtrainD = zip(*trainD)

mean, eigenvec, thetas = train(mtrainD)
print "Mean: ", mean
print "eigenvec: \n", eigenvec
print "thetas: \n", thetas
# exit(-1)
# thetaSubmit = submit(mtxLst[0], mean, eigenvec)
# print "\n"
# print "M0 delta target", np.absolute(np.sum(thetas[0] - thetaSubmit))
# print "M1 delta", np.absolute(np.sum(thetas[1] - thetaSubmit))
# print "M2 delta", np.absolute(np.sum(thetas[2] - thetaSubmit))
# print "M3 delta", np.absolute(np.sum(thetas[3] - thetaSubmit))
# print "M4 delta", np.absolute(np.sum(thetas[4] - thetaSubmit))
# print "M5 delta", np.absolute(np.sum(thetas[5] - thetaSubmit))
# print "M6 delta", np.absolute(np.sum(thetas[6] - thetaSubmit))
for idx, data in enumerate(validD):
    filename, mtx = data
    thetaSubmit = subm.submit(mtx, mean, eigenvec)
    matchIdx = subm.compare(thetaSubmit, thetas)
    print filename, " --> ", ftrainD[matchIdx]
