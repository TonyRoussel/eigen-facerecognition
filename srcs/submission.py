from training import gradDescent
from constants import learRate
import numpy as np

def submit(mtx, mean, eigenvec):
    mtxflat = mtx.flatten()
    mtxflat = mtxflat[:, np.newaxis] - mean[:, np.newaxis]
#     eigenvecOne = np.insert(eigenvec, 0, np.ones(np.shape(eigenvec)[0]), axis=1)
    theta = np.random.rand(np.shape(eigenvec)[1])
    theta = gradDescent(eigenvec, np.matrix(mtxflat).transpose(), np.matrix(theta).transpose(), learRate)
    return theta

