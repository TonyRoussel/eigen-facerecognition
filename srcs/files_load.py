from scipy import misc
import constants as cst
import os
import time

usleep = lambda x: time.sleep(x/1000000.0)

def loadmatrixs(path):
    matrixs = []
    sze = len(os.listdir(path))
    if sze > cst.fileloadLimit:
        sze = cst.fileloadLimit
    for i, filename in enumerate(os.listdir(path)):
        if not filename.endswith(cst.img_extension):
            continue
        img = misc.imread(path + filename)
        usleep(2000)
        matrixs.append((filename, img))
        print "file loading: ", i, " / ", sze #####
        if i == cst.fileloadLimit:
            break
    return matrixs

def split_list(mylist, percent):
    lft_sze = int(percent / 100. * len(mylist))
    rht_sze = len(mylist) - lft_sze
    lft = mylist[:lft_sze]
    rht = mylist[-rht_sze:]
    return (lft, rht)

if __name__ == "__main__":
    import sys
    fm = loadmatrixs(sys.argv[1])
    train, valid = split_list(fm, 10)
    ftrain, vtrain = zip(*train)
    fvalid, vvalid = zip(*valid)
    print fvalid
