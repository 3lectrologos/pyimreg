import os
import pickle
import math
import progressbar as pb
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
import imreg
import chist


IMG_DIR = 'zubud_small'
CLUSTERS = 30
FILES_PER_CLUSTER = 5

files = os.listdir(IMG_DIR)
files.sort()
if CLUSTERS == None:
    CLUSTERS = int(math.floor(len(files)/float(FILES_PER_CLUSTER)))
images = [cv2.imread(os.path.join(IMG_DIR, f)) 
          for f in files[:CLUSTERS*FILES_PER_CLUSTER]]
print '# of images =', len(images)

widgets = [pb.Percentage(), ' ', pb.ETA(), ' ',
           pb.Bar(left='|', right='|', marker='-')]
bar = pb.ProgressBar(widgets=widgets, maxval=len(images)**2)
it = 0
nm = np.zeros((len(images), len(images)))
cd = np.zeros((len(images), len(images)))
for i in range(len(images)):
    for j in range(len(images)):
        nm[i,j] = imreg.numMatches(images[i], images[j])
        cd[i,j] = chist.distance(images[i], images[j])
        #print (i, j), '->', 'matches:', nm[i,j], ', dist:', cd[i,j]
        it = it + 1
        bar.update(it)
bar.finish()

with open('tmp.pickle', 'w') as fout:
    pickle.dump((nm, cd), fout)

plt.plot(nm, cd, 'ro')
for i in range(CLUSTERS):
    lo = i*FILES_PER_CLUSTER
    hi = (i+1)*FILES_PER_CLUSTER
    plt.plot(nm[lo:hi, lo:hi], cd[lo:hi, lo:hi], 'go')
plt.show()
