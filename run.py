import os
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


CLUSTERS = 200
TMP_FILE = 'tmp.pickle'

if os.path.isfile(TMP_FILE):
    with open('tmp.pickle', 'r') as fin:
        (nm, cd) = pickle.load(fin)
else:
    (nm, cd) = util.get_scores(CLUSTERS, 'both', hbins=64, rand=True)
    with open('tmp.pickle', 'w') as fout:
        pickle.dump((nm, cd), fout)

#np.savetxt('img_matches.txt', nm, fmt="%d")
        
(images, files) = util.get_images(CLUSTERS)
print [(files[i], files[j]) for i in range(1000) for j in range(1000)
       if (nm[i, j] > 10 and
           j not in range(i - i % util.FILES_PER_CLUSTER,
                          i + (util.FILES_PER_CLUSTER - i % util.FILES_PER_CLUSTER)))]
        
plt.plot(nm, cd, 'ro')
for i in range(CLUSTERS):
    lo = i*util.FILES_PER_CLUSTER
    hi = (i+1)*util.FILES_PER_CLUSTER
    plt.plot(nm[lo:hi, lo:hi], cd[lo:hi, lo:hi], 'go')
plt.show()
