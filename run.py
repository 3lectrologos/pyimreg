import os
import os.path
import pickle
import math
import progressbar as pb
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
import imreg
import chist


IMG_DIR = 'zubud'
CLUSTERS = 30
FILES_PER_CLUSTER = 5
TMP_FILE = 'tmp.pickle'


def get_images(clusters):
    files = os.listdir(IMG_DIR)
    files.sort()
    if clusters == None:
        clusters = int(math.floor(len(files)/float(FILES_PER_CLUSTER)))
    images = [cv2.imread(os.path.join(IMG_DIR, f))
              for f in files[:clusters*FILES_PER_CLUSTER]]
    return (images, files)
    
def get_scores():
    (images, _) = get_images(CLUSTERS)
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
    return (nm, cd)

if os.path.isfile(TMP_FILE):
    with open('tmp.pickle', 'r') as fin:
        (nm, cd) = pickle.load(fin)
else:
    (nm, cd) = get_scores()
    with open('tmp.pickle', 'w') as fout:
        pickle.dump((nm, cd), fout)
        
(images, files) = get_images(CLUSTERS)
print [(files[i], files[j]) for i in range(100) for j in range(100)
       if nm[i, j] == 100 and i >= j + 5]
        
plt.plot(nm, cd, 'ro')
for i in range(CLUSTERS):
    lo = i*FILES_PER_CLUSTER
    hi = (i+1)*FILES_PER_CLUSTER
    plt.plot(nm[lo:hi, lo:hi], cd[lo:hi, lo:hi], 'go')
plt.show()
