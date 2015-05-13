from training import gradDescent
import numpy as np

def submit(mtx, mean, eigenvec):
    print "mtx shape: ", np.shape(mtx) ###########
    print "mean: ", mean ###########
    print "eigenvec shape: ", np.shape(eigenvec) ###########
    mtxflat = mtx.flatten()
    mtxflat = mtxflat[:, np.newaxis] - mean[:, np.newaxis]
    eigenvecOne = np.insert(eigenvec, 0, np.ones(np.shape(eigenvec)[0]), axis=1)
    theta = np.random.rand(np.shape(eigenvecOne)[1])
    print "eigenvecOne shape: ", np.shape(eigenvecOne) ###########
    print "np.matrix(mtxflat) shape: ", np.shape(np.matrix(mtxflat)) ####
    theta = gradDescent(eigenvecOne, np.matrix(mtxflat), np.matrix(theta).transpose(), 1e-4, 200)
    return theta

