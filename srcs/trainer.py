import numpy as np
# from constants import mtxLst
import training as trn
from qr import qr

# print "toCompute Matrix: \n", mtxLst
# M = trn.concatMatrix(mtxLst)
# print "post concatenation Matrix: \n", M
# M = M - M.mean()
# print "post mean sub Matrix: \n", M
# Mtld = np.dot(M.transpose(), M)
# print "M tilde: \n",Mtld
# eigenval, eigenvec = qr(Mtld)
# print "eigenval: \n", eigenval
# print "eigenvec: \n", eigenvec
# eigenvec = trn.extractEigenvecOnVal(eigenval, eigenvec)
# print "eigenvec post extract: \n", eigenvec
# eigenvec = trn.reconstructVector(Mtld, eigenvec)
# print "eigenvec post reconstruct: \n", eigenvec
# # eigenvec = np.row_stack((np.ones(np.shape(eigenvec)[1]), eigenvec))
# eigenvec = np.insert(eigenvec, 0, np.ones(np.shape(eigenvec)[0]), axis=1)
# print "eigenvec post ones: \n", eigenvec
# # eigenvec = eigenvec.transpose()
# # print "eigenvec post transpose: \n", eigenvec
# imgA = Mtld.transpose()[0]
# imgA = imgA.transpose()
# theta = np.random.rand(np.shape(eigenvec)[1])
# print "x: \n", eigenvec
# print "theta: \n", theta
# print "y: \n", imgA
# theta = trn.gradDescent(eigenvec, np.matrix(imgA), np.matrix(theta).transpose(), 1e-16, 100)
# print "eigenvec : \n", eigenvec
# print "final theta: \n", theta
# print "imgA : \n", imgA 
# print "eigenvec * theta: \n", np.dot(eigenvec, theta)

def train(mtxLst):
    thetas = list()
    M = trn.concatMatrix(mtxLst)
    Mmean = M.mean()
    M = M - Mmean
    Mtld = np.dot(M.transpose(), M)
    n = np.shape(Mtld)[1]
    eigenval, eigenvec = qr(Mtld)
    eigenvec = trn.extractEigenvecOnVal(eigenval, eigenvec)
    eigenvec = trn.reconstructVector(Mtld, eigenvec)
    eigenvec = np.insert(eigenvec, 0, np.ones(np.shape(eigenvec)[0]), axis=1)
    for i in range(n):
        img = Mtld.transpose()[i]
        theta = np.random.rand(np.shape(eigenvec)[1])
        theta = trn.gradDescent(eigenvec, np.matrix(img), np.matrix(theta).transpose(), 1e-16, 150)
        thetas.append(theta)
    return (Mmean, eigenvec, thetas)
