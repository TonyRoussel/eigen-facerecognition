from trainer import train
from submission import submit
from constants import mtxLst
from constants import zinedine03
from constants import meryl03
import numpy as np

mean, eigenvec, thetas = train(mtxLst)
print "Mean: ", mean
print "eigenvec: \n", eigenvec
print "thetas: \n", thetas

thetaSubmit = submit(mtxLst[0], mean, eigenvec)
print "\n"
print "M0 delta target", np.absolute(np.sum(thetas[0] - thetaSubmit))
print "M1 delta", np.absolute(np.sum(thetas[1] - thetaSubmit))
print "M2 delta", np.absolute(np.sum(thetas[2] - thetaSubmit))
print "M3 delta", np.absolute(np.sum(thetas[3] - thetaSubmit))
print "M4 delta", np.absolute(np.sum(thetas[4] - thetaSubmit))
print "M5 delta", np.absolute(np.sum(thetas[5] - thetaSubmit))
# print "M6 delta", np.absolute(np.sum(thetas[6] - thetaSubmit))

thetaSubmit = submit(mtxLst[3], mean, eigenvec)
print "\n"
print "M0 delta", np.absolute(np.sum(thetas[0] - thetaSubmit))
print "M1 delta", np.absolute(np.sum(thetas[1] - thetaSubmit))
print "M2 delta", np.absolute(np.sum(thetas[2] - thetaSubmit))
print "M3 delta target", np.absolute(np.sum(thetas[3] - thetaSubmit))
print "M4 delta", np.absolute(np.sum(thetas[4] - thetaSubmit))
print "M5 delta", np.absolute(np.sum(thetas[5] - thetaSubmit))
# print "M6 delta", np.absolute(np.sum(thetas[6] - thetaSubmit))

thetaSubmit = submit(zinedine03, mean, eigenvec)
print "\n"
print "M0 delta", np.absolute(np.sum(thetas[0] - thetaSubmit))
print "M1 delta", np.absolute(np.sum(thetas[1] - thetaSubmit))
print "M2 delta", np.absolute(np.sum(thetas[2] - thetaSubmit))
print "M3 delta target", np.absolute(np.sum(thetas[3] - thetaSubmit))
print "M4 delta target", np.absolute(np.sum(thetas[4] - thetaSubmit))
print "M5 delta target", np.absolute(np.sum(thetas[5] - thetaSubmit))
# print "M6 delta", np.absolute(np.sum(thetas[6] - thetaSubmit))

thetaSubmit = submit(meryl03, mean, eigenvec)
print "\n"
print "M0 delta target", np.absolute(np.sum(thetas[0] - thetaSubmit))
print "M1 delta target", np.absolute(np.sum(thetas[1] - thetaSubmit))
print "M2 delta target", np.absolute(np.sum(thetas[2] - thetaSubmit))
print "M3 delta", np.absolute(np.sum(thetas[3] - thetaSubmit))
print "M4 delta", np.absolute(np.sum(thetas[4] - thetaSubmit))
print "M5 delta", np.absolute(np.sum(thetas[5] - thetaSubmit))
# print "M6 delta", np.absolute(np.sum(thetas[6] - thetaSubmit))
