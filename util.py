import os
import random
import numpy as np
import cv2
import progressbar as pb
import chist
import imreg


IMG_DIR = 'zubud'
FILES_PER_CLUSTER = 5

def get_images(clusters):
    files = os.listdir(IMG_DIR)
    files.sort()
    if clusters == None:
        clusters = int(math.floor(len(files)/float(FILES_PER_CLUSTER)))
    images = [cv2.imread(os.path.join(IMG_DIR, f))
              for f in files[:clusters*FILES_PER_CLUSTER]]
    return (images, files)

def get_scores(clusters, mode='both', hbins=64, rand=False):
    if mode == 'both' or mode == 'match':
        run_match = True
    else:
        run_match = False
    if mode == 'both' or mode == 'hist':
        run_hist = True
    else:
        run_hist = False
    (images, _) = get_images(clusters)
    print '# of images =', len(images)
    widgets = [pb.Percentage(), ' ', pb.ETA(), ' ',
               pb.Bar(left='|', right='|', marker='-')]
    bar = pb.ProgressBar(widgets=widgets, maxval=len(images)**2)
    it = 0
    if run_match:
        nm = np.zeros((len(images), len(images)))
    if run_hist:
        cd = np.zeros((len(images), len(images)))
    hcache = dict()
    for i in range(len(images)):
        if rand:
            mod = i % FILES_PER_CLUSTER
            jrange = range(i - mod, i + (FILES_PER_CLUSTER - mod))
            jrand = random.sample(set(range(len(images))) - set(jrange), 10)
            print jrange
            print jrand
            jrange = jrange + jrand
        else:
            jrange = range(len(images))
        for j in jrange:
            print (i, j)
            try:
                hi = hcache[i]
            except KeyError:
                hi = chist.getHistogram(images[i], hbins)
                hcache[i] = hi
            try:
                hj = hcache[j]
            except KeyError:
                hj = chist.getHistogram(images[j], hbins)
                hcache[j] = hj
            if run_match:
                nm[i, j] = imreg.numMatches(images[i], images[j])
            if run_hist:
                cd[i, j] = chist.hdistance(hi, hj)
            it = it + 1
            bar.update(it)
    bar.finish()
    if run_match and run_hist:
        return (nm, cd)
    elif run_match:
        return nm
    elif run_hist:
        return cd
