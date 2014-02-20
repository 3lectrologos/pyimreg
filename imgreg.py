import numpy as np
import cv2

def filterMatchesHomography(kp1, kp2, matches):
    points1 = np.array([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    points2 = np.array([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 6.0)
    mask = mask.ravel().tolist()
    goodMatches = [m for (i, m) in enumerate(matches) if mask[i] == 1]
    return (goodMatches, H)

def getLargeImage(img1, img2, X_OFF=100):
    h1 = img1.shape[0]
    h2 = img2.shape[0]
    w1 = img1.shape[1]
    w2 = img2.shape[1]
    h = max(h1, h2)
    w = max(w1, w2)
    img = np.zeros((h, w1 + w2 + X_OFF), dtype=img1.dtype)
    img[:h1,:w1] = img1
    img[-h2:,-w2:] = img2
    return img

def tint(x):
    return (int(x[0]), int(x[1]))

def getTransform(img1, img2):
    h1 = img1.shape[0]
    h2 = img2.shape[0]
    h = max(h1, h2)
    w1 = img1.shape[1]
    return lambda x: tint((x[0] + w1 + 100, x[1] + h - h2))

def plotKeypoints(keypoints, transform=None, color=(255, 0, 0)):
    if transform == None:
        transform = lambda x: x
    for kp in keypoints:
        p = kp.pt
        cv2.circle(img, tint(transform(p)), 3, color=color, thickness=1)

def plotMatches(keypoints, matches, transform, color=(0, 255, 0)):
    for m in matches:
        p1 = keypoints[0][m.queryIdx].pt
        p2 = keypoints[1][m.trainIdx].pt  
        cv2.line(img, tint(p1), transform(p2), color=color)

def detectSift(img1, img2):
    detector = cv2.SIFT()
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)
    return (kp1, kp2, des1, des2)

def detectSurf(img1, img2):
    detector = cv2.SURF()
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)
    return (kp1, kp2, des1, des2)

def detectOrb(img1, img2):
    detector = cv2.ORB(30000, 1.2, 8, 31, 0, 2, cv2.ORB_HARRIS_SCORE, 31)
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)
    return (kp1, kp2, des1, des2)

def detectBriskFreak(img1, img2):
    detector = cv2.ORB(10000, 1.2, 8, 31, 0, 2, cv2.ORB_HARRIS_SCORE, 31)
    #detector = cv2.FeatureDetector_create('ORB')
    descriptor = cv2.DescriptorExtractor_create('FREAK')
    kp1 = detector.detect(img1, None)
    kp2 = detector.detect(img2, None)
    kp1, des1 = descriptor.compute(img1, kp1)
    kp2, des2 = descriptor.compute(img2, kp2)
    return (kp1, kp2, des1, des2)

def detectSurfOrb(img1, img2):
    detector = cv2.FeatureDetector_create('SURF')
    descriptor = cv2.DescriptorExtractor_create('ORB')
    kp1 = detector.detect(img1, None)
    kp2 = detector.detect(img2, None)
    kp1, des1 = descriptor.compute(img1, kp1)
    kp2, des2 = descriptor.compute(img2, kp2)
    return (kp1, kp2, des1, des2)

def matchFlann(des1, des2):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=64)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    return matches

def matchBf(des1, des2, norm=cv2.NORM_HAMMING):
    bf = cv2.BFMatcher(norm, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    goodMatches = [m for (m, n) in matches if m.distance < 0.7*n.distance]
    return (matches, goodMatches)

#FILES = ('rsz_1cab.jpg', 'cab_low.jpg')
FILES = ('dominos2.jpg', 'dominos1.jpg')
#FILES = ('cab_new.jpg', 'cab_low.jpg')
#FILES = ('cab_night.jpg', 'rsz_1cab.jpg')
#FILES = ('cab_night.jpg', 'cab_low.jpg')
img1 = cv2.imread(FILES[0], cv2.IMREAD_GRAYSCALE)
img1 = cv2.resize(img1, (640, 360))
img2 = cv2.imread(FILES[1], cv2.IMREAD_GRAYSCALE)
t = getTransform(img1, img2)
(kp1, kp2, des1, des2) = detectOrb(img1, img2)
(matches, goodMatches) = matchBf(des1, des2, norm=cv2.NORM_HAMMING)
(goodMatchesHom, H) = filterMatchesHomography(kp1, kp2, goodMatches)

print 'keypoints = ', len(kp1), ' -- ', len(kp2)
print 'Good: ', len(goodMatches), ' out of ', len(matches)
print 'Homography: ', len(goodMatchesHom), ' out of ', len(goodMatches)
print H

img = getLargeImage(img1, img2)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
plotKeypoints(kp1)
plotKeypoints(kp2, t)
plotMatches((kp1, kp2), goodMatchesHom, t)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
