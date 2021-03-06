#!/usr/bin/env python

import argparse
import os.path
import pickle
import math

import progressbar as pb
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image

import util
import imreg
import chist


parser = argparse.ArgumentParser()
parser.add_argument('-c', default=10, type=int, dest='nclusters')
parser.add_argument('-plot', action='store_true')
args = parser.parse_args()
nclusters = args.nclusters
fout = 'score_' + str(nclusters) + '.pickle'

if os.path.isfile(fout):
    with open(fout, 'r') as fin:
        (nm, cd) = pickle.load(fin)
else:
    (nm, cd) = util.get_scores(nclusters, 'both', hbins=64, rand=True)
    with open(fout, 'w') as fout:
        pickle.dump((nm, cd), fout)

(images, files) = util.get_images(nclusters)
#print [(files[i], files[j]) for i in range(1000) for j in range(1000)
#       if (nm[i, j] > 15 and not util.same_cluster(i, j))]

x = []
y = []
for i in range(nm.shape[0]):
    for j in range(nm.shape[1]):
        if nm[i, j] != 0:
            x.append((nm[i, j], cd[i, j]))
            if util.same_cluster(i, j):
                y.append(1)
            else:
                y.append(-1)
x = np.array(x)
y = np.array(y)
util.train_classifier(x, y)

if args.plot:
    plt.plot(x[y==-1, 0], x[y==-1, 1], 'ro')
    plt.plot(x[y==1, 0], x[y==1, 1], 'go')
    plt.show()
