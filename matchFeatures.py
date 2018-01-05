import cv2
import numpy as np

fullImg = cv2.imread("images/ryan.jpg")
image = cv2.imread("images/ryan2.jpg")

ratio=0.75
reprojThresh=4.0

# create a SIFT object
sift = cv2.xfeatures2d.SIFT_create()
# find keypoints and compute descriptors
(kps1,descriptors1) = sift.detectAndCompute(image,None)
(kps2,descriptors2) = sift.detectAndCompute(fullImg,None)

matcher = cv2.DescriptorMatcher_create("BruteForce")
rawMatches = matcher.knnMatch(descriptors1,descriptors2,2)
matches = []

for m,n in rawMatches:
    # ensure the distance is within a certain ratio of each other (i.e. Lowe's ratio test)
    if m.distance < 0.7 * n.distance:
        matches.append(m)

# compute homography which requires at least 4 matches
if len(matches) > 4:
    # construct the two sets of points
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # compute the homography between the two sets of points
    # we want to find the transformation matrix between the each keypoint from A and B so that we know
    # how to transform the second image to align with the first image
    (H, status) = cv2.findHomography(pts1, pts1, cv2.RANSAC,reprojThresh)
    matchesMask = status.ravel().tolist()

    h, w = image.shape[:2]
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H)

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img3 = cv2.drawMatches(image, kps1, fullImg, kps2, matches, None, **draw_params)
    cv2.imshow("Matches",img3)
    cv2.waitKey(0)

else:
    print("Not enough matches found!")



