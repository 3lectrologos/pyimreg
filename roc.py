import numpy as np
import matplotlib.pylab as plt
import scipy.linalg
import sklearn.metrics

import util


CLUSTERS = 200
XINTERC = 0.001

bmat = np.ones((util.FILES_PER_CLUSTER, util.FILES_PER_CLUSTER), dtype = 'bool')
blocks = [bmat for i in range(CLUSTERS)]
true = scipy.linalg.block_diag(*blocks)
hbins_array = [16, 25, 32, 50, 64, 80, 100, 128]
aucs = []
for hbins in hbins_array:
    scores = -util.get_scores(CLUSTERS, 'hist', hbins=hbins)
    fpr, tpr, thresh = sklearn.metrics.roc_curve(true.flatten(), scores.flatten())
    print np.interp(XINTERC, fpr, tpr)
#    plt.plot(fpr, tpr)
#    plt.show()
#    aucs.append(sklearn.metrics.roc_auc_score(true.flatten(), scores.flatten()))
    aucs.append(np.interp(XINTERC, fpr, tpr))
    print (hbins, aucs[-1])

plt.plot(hbins_array, aucs)
plt.show()
