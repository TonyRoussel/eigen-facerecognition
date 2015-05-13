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
    Mmean = M.mean(axis=1)
    print "M :\n", M ##########
    print "Mmean :\n", Mmean ##########
    M -= Mmean[:, np.newaxis]
    Mtld = np.dot(M.transpose(), M)
    n = np.shape(Mtld)[1]
    eigenval, eigenvec = qr(Mtld)
    eigenvec = trn.extractEigenvecOnVal(eigenval, eigenvec)
    eigenvec = trn.reconstructVector(Mtld, eigenvec)
    eigenvecOne = np.insert(eigenvec, 0, np.ones(np.shape(eigenvec)[0]), axis=1)
    for i in range(n):
        img = Mtld.transpose()[i]
        theta = np.random.rand(np.shape(eigenvecOne)[1])
        print "eigenvecOne shape: ", np.shape(eigenvecOne) ###########
        print "np.matrix(img) shape: ", np.shape(np.matrix(img)) ####
        theta = trn.gradDescent(eigenvecOne, np.matrix(img), np.matrix(theta).transpose(), 1e-4, 200)
        thetas.append(theta)
    print "Mtld[0]: \n", Mtld[0]
    print "eigenvec * theta: \n", np.dot(eigenvecOne, thetas[0])
    return (Mmean, eigenvec, thetas)
