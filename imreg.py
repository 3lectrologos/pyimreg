import numpy as np
import numpy.linalg
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

def detectOrb(img):
    detector = cv2.ORB(30000, 1.2, 8, 31, 0, 4, cv2.ORB_HARRIS_SCORE, 31)
    kp, des = detector.detectAndCompute(img, None)
    return (kp, des)

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
    goodMatches = [m for (m, n) in matches if m.distance < 0.75*n.distance]
#    matches = [m[0] for m in matches if m != []]
    return (matches, goodMatches)

def getTransformedBox(img, H):
    (h, w) = img.shape
    corners = np.array([[0, 0], [h, 0], [h, w], [0, w]], dtype='float32')
    corners = np.array([corners])
    return cv2.perspectiveTransform(corners, H)

def numMatches(img1, img2, cache1=None, cache2=None):
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    (kp1, des1) = detectOrb(img1)
    (kp2, des2) = detectOrb(img2)
    (matches, goodMatches) = matchBf(des1, des2, norm=cv2.NORM_HAMMING)
    (goodMatchesHom, H) = filterMatchesHomography(kp1, kp2, goodMatches)
    if numpy.linalg.det(H) < 0.0001:
        return 1
    trcorners = getTransformedBox(img1, H)
    if not cv2.isContourConvex(trcorners):
        return 1
    return min(100, len(goodMatchesHom))

if __name__ == '__main__':
    FILES = ('zubud/object0012.view01.png', 'zubud/object0001.view01.png')
    img1 = cv2.imread(FILES[0], cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(FILES[1], cv2.IMREAD_GRAYSCALE)
    #img2 = cv2.resize(img2, (640, 360))
    t = getTransform(img1, img2)
    (kp1, des1) = detectOrb(img1)
    (kp2, des2) = detectOrb(img2)
    (matches, goodMatches) = matchBf(des1, des2, norm=cv2.NORM_HAMMING)
    (goodMatchesHom, H) = filterMatchesHomography(kp1, kp2, goodMatches)
    trcorners = getTransformedBox(img1, H)

    print 'Keypoints =', len(kp1), '--', len(kp2)
    print 'Good:', len(goodMatches), 'out of', len(matches)
    print 'Homography:', len(goodMatchesHom), 'out of', len(goodMatches)
    print H
    print 'det(H) =', numpy.linalg.det(H)
    print 'Convex?', cv2.isContourConvex(trcorners)

    img = getLargeImage(img1, img2)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    plotKeypoints(kp1)
    plotKeypoints(kp2, t)
    plotMatches((kp1, kp2), goodMatchesHom, t)
    for i in range(4):
        p1 = t(trcorners[0,i,:])
        p2 = t(trcorners[0,(i+1)%4,:])
        cv2.line(img, p1, p2, thickness=3, color=(0, 0, 255))
    
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
