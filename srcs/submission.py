from training import gradDescent
from constants import learRate
import numpy as np

def submit(mtx, mean, eigenvec):
    mtxflat = mtx.flatten()
    mtxflat = mtxflat[:, np.newaxis] - mean[:, np.newaxis]
    eigenvecOne = np.insert(eigenvec, 0, np.ones(np.shape(eigenvec)[0]), axis=1)
    theta = np.random.rand(np.shape(eigenvecOne)[1])
    # print "eigenvecOne :", type(eigenvecOne), np.shape(eigenvecOne) ###
    # print "np.matrix(mtxflat).transpose() :", type(np.matrix(mtxflat).transpose()), np.shape(np.matrix(mtxflat).transpose()) ###
    # print "np.matrix(theta).transpose() :", type(np.matrix(theta).transpose()), np.shape(np.matrix(theta).transpose()) ###
    theta = gradDescent(eigenvecOne, np.matrix(mtxflat).transpose(), np.matrix(theta).transpose(), learRate)
    return theta

