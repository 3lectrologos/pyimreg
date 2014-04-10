import numpy as np
import scipy.spatial.distance
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.image

def sim(img1, img2):
    pass

def getHistogram(img, bins=64):
    img = matplotlib.colors.rgb_to_hsv(img)
    (hist, _) = np.histogram(img[:,:,0].flatten(), bins=bins, density=True)
    return hist

def distance(img1, img2):
    h1 = getHistogram(img1)
    h2 = getHistogram(img2)
    return scipy.spatial.distance.cityblock(h1, h2)

if __name__ == '__main__':
    FILES = ('img/dominos1.jpg',
             'zubud_small/object0003.view05.png',
             'zubud_small/object0003.view03.png')
    img1 = matplotlib.image.imread(FILES[1])
    img2 = matplotlib.image.imread(FILES[2])
    h1 = getHistogram(img1)
    h2 = getHistogram(img2)
    dist = scipy.spatial.distance.cityblock(h1, h2)
    print dist
#    plt.colorbar()
#    plt.hsv()
    plt.show()
    #img2 = cv2.imread(FILES[1])
    #cv2.cvtColor(img2, cv2.COLOR_RBG2HSV_FULL)
    #hist = cv2.calcHist(img1,
    #                    channels=0,
    #                    mask=0,
    #                    histSize=64,
    #                    ranges=[0, 128])
    
