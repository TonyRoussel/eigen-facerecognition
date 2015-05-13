from trainer import train
from submission import submit
from constants import mtxLst
import numpy as np

mean, eigenvec, thetas = train(mtxLst)
print "Mean: ", mean
print "eigenvec: \n", eigenvec
print "thetas: \n", thetas
print submit(mtxLst[0], mean, eigenvec)
