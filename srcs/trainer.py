import numpy as np
from constants import mtxLst
import training as trn
from qr import qr

print "toCompute Matrix: \n", mtxLst
M = trn.concatMatrix(mtxLst)
print "post concatenation Matrix: \n", M
M = M - M.mean()
print "post mean sub Matrix: \n", M
Mtld = np.dot(M.transpose(), M)
print "M tilde: \n",Mtld
eigenval, eigenvec = qr(Mtld)
print "eigenval: \n", eigenval
print "eigenvec: \n", eigenvec
eigenvec = trn.extractEigenvecOnVal(eigenval, eigenvec)
print "eigenvec post extract: \n", eigenvec
eigenvec = trn.reconstructVector(Mtld, eigenvec)
print "eigenvec post reconstruct: \n", eigenvec
# eigenvec = np.row_stack((np.ones(np.shape(eigenvec)[1]), eigenvec))
eigenvec = np.insert(eigenvec, 0, np.ones(np.shape(eigenvec)[0]), axis=1)
print "eigenvec post ones: \n", eigenvec
# eigenvec = eigenvec.transpose()
# print "eigenvec post transpose: \n", eigenvec
imgA = Mtld.transpose()[0]
imgA = imgA.transpose()
theta = np.ones(np.shape(eigenvec)[0])
print "x: \n", np.shape(eigenvec)
print "theta: \n", np.shape(theta)
print "y: \n", np.shape(imgA)
theta = trn.gradDescent(eigenvec, imgA, theta, 0.0001, 10000)
print "final theta: \n", theta
print np.dot(theta, eigenvec)
