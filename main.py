# This file contains the driver code for Panorama Stitching
from helpers import *

image0 = cv2.imread("inputImages/I2/2_1.JPG")
image1 = cv2.imread("inputImages/I2/2_2.JPG")

# Using ORB
(keypoints0, features0) = extract_features_keypoints(image0)
(keypoints1, features1) = extract_features_keypoints(image1)

# Using KNN Matching, with default K = 2
all_matches = match_features(features0, features1)

# Using Lowe's Ratio, with default ratio = 0.8
valid_matches = valid_matches(all_matches)

keypoint_matrix = correspondence_matrix(
    valid_matches, keypoints0, keypoints1)

sigma = 1
threshold = np.sqrt(5.99) * sigma

H, inliers = get_homography(keypoint_matrix, threshold)