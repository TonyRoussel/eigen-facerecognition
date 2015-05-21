import numpy as np
import training as trn
from qr import qr
from constants import learRate
from constants import maxIteration
import matplotlib.pyplot as plt

def train(mtxLst):
    thetas = list()
    M = trn.concatMatrix(mtxLst)
    Mmean = M.mean(axis=1)
    # plt.imshow(np.reshape(Mmean, np.shape(mtxLst[0])))
    # plt.gray()
    # plt.show()
    M -= Mmean[:, np.newaxis]
    Mtld = np.dot(M.transpose(), M)
    n = np.shape(Mtld)[1]
    # eigenval, eigenvec = qr(Mtld, 200)
    eigenval, eigenvec = np.linalg.eig(Mtld)
    # eigenvec = trn.extractEigenvecOnVal(eigenval, eigenvec, -2)
    eigenvec = trn.reconstructVector(M, eigenvec)
    # for i in range(np.shape(eigenvec.transpose())[0]):
    #     plt.imshow(np.reshape(eigenvec.transpose()[i], np.shape(mtxLst[0])))
    #     plt.gray()
    #     plt.show()
    # eigenvec = np.insert(eigenvec, 0, np.ones(np.shape(eigenvec)[0]), axis=1)
    print "eigenvec shape: ", np.shape(eigenvec)
    for i in range(n):
        img = M.transpose()[i]
        theta = np.ones(np.shape(eigenvec)[1])
        theta = trn.gradDescent(eigenvec, np.matrix(img), np.matrix(theta).transpose(), learRate, maxIteration)
        print "Descent terminated: ", i, " / ", n - 1  ######
        thetas.append(theta)
    # plt.imshow(np.reshape(np.dot(eigenvec, thetas[0]), np.shape(mtxLst[0])))
    # plt.gray()
    # plt.show()
    return (Mmean, eigenvec, thetas)
