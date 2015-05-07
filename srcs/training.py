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

def gradientDescent(x, y, theta, alpha, m, numIterations):
    xTrans = x.transpose()
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        gradient = np.dot(xTrans, loss) / m
        # update
        theta = theta - alpha * gradient
    return theta
