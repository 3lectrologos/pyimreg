import numpy as np
import scipy.spatial.distance
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.image


def getHistogram(img, bins):
    img = matplotlib.colors.rgb_to_hsv(img)
    (hist, _) = np.histogram(img[:,:,0].flatten(), bins=bins, density=True)
    return hist

def hdistance(h1, h2):
    return scipy.spatial.distance.cityblock(h1, h2)

def distance(img1, img2, bins=64):
    h1 = getHistogram(img1, bins)
    h2 = getHistogram(img2, bins)
    return hdistance(h1, h2)

if __name__ == '__main__':
    FILES = ('img/cab_new.jpg', 'img/cab_night.jpg')
    img1 = matplotlib.image.imread(FILES[0])
    img2 = matplotlib.image.imread(FILES[1])
    h1 = getHistogram(img1, 64)
    h2 = getHistogram(img2, 64)
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
    
