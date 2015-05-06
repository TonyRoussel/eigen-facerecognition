import numpy as np
from constants import mtxLst
import training as trn
from qr import qr

print mtxLst
M = trn.concatMatrix(mtxLst) ##########
print M
M = M - M.mean()
print M
Mtld = np.dot(M.transpose(), M)
print Mtld
eigenval, eigenvec = qr(Mtld)
eigenvec = trn.extractEigenvecOnVal(eigenval, eigenvec)
eigenvec = trn.reconstructVector(Mtld, eigenvec)
print "eigenvec: \n", eigenvec
