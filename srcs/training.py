import numpy as np

def concatMatrix(mtxLst):
    mtx = np.array(())
    flatLst = []
    for m in mtxLst:
        flatLst.append(m.flatten())
    mtx = np.vstack(flatLst)
    return mtx.transpose()

def extractEigenvecOnVal(eigval, eigvec, threshold = 1):
    delIdx = np.where(eigval < threshold)[0]
    return np.delete(eigvec, delIdx, axis=1)

def reconstructVector(M, eigvec):
    eigvecT = eigvec.transpose()
    newmatrix = np.empty(np.shape(eigvecT))
    for idx, vec in enumerate(eigvecT):
        newvec = np.dot(M, vec.transpose())
        newmatrix[idx] = newvec
    return newmatrix.transpose()

def computeCostMulti(X, y, theta):
    H = np.dot(X, theta)
    diff = H - y
    diff = np.power(diff, 2)
    sdiff = np.sum(diff, axis=0)
    return (sdiff / (2 * (np.shape(y)[0])))

def gradDescent(X, y, theta, alpha, numIter = 100):
    m = np.shape(y)[0]
    for i in range(numIter):
        H = np.dot(X, theta)
        diff = H - y
        diffm = np.multiply(X, diff)
        sdiff = np.sum(diffm, axis=0).transpose()
        sigma = np.multiply(1. / m, sdiff)
        theta = theta - alpha * sigma
        print "Cost: ", computeCostMulti(X, y, theta)
    return theta
