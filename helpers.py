# This file contains all the helper functions for Panorama Stitching

import cv2
import numpy as np
import random
# Part 1: Detect, extract, and match features


def extract_features_keypoints(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    (Keypoints, features) = orb.detectAndCompute(image, None)
    return (Keypoints, features)


def match_features(features1, features2, K=2):
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(features1, features2, K)
    return matches


def valid_matches(matches, lowe_ratio=0.8):
    valid_matches = []

    for k_match in matches:
        if len(k_match) == 2 and k_match[0].distance < k_match[1].distance * lowe_ratio:
            # valid_matches.append((k_match[0].trainIdx, k_match[0].queryIdx))
            valid_matches.append(k_match[0])
    return valid_matches


def correspondence_matrix(valid_matches, keypoints0, keypoints1):
    keypoints = []

    for match in valid_matches:
        (x0, y0) = keypoints0[match.queryIdx].pt
        (x1, y1) = keypoints1[match.trainIdx].pt
        keypoints.append([x0, y0, x1, y1])

    keypoint_matrix = np.matrix(keypoints)
    return keypoint_matrix

def calcH(sample):
    return 0

def geometricDistance(matrixList, H):
    return 0

def get_homography(matrix, threshold, n_iter=500):
    inliers = []
    n = len(matrix)
    for i in range(n_iter):
        random1 = matrix[random.randrange(0, n)]
        random2 = matrix[random.randrange(0, n)]
        random3 = matrix[random.randrange(0, n)]
        random4 = matrix[random.randrange(0, n)]
        random_sample = np.vstack((random1, random2, random3, random4))

        H = calcH(random_sample)
        iteration_inliers = []

        for i in range(n):
            d = geometricDistance(matrix[i], H)
            if d < 5:
                iteration_inliers.append(matrix[i])

        if len(iteration_inliers) > len(inliers):
            inliers = iteration_inliers
            finalH = H

        if len(inliers) > len(matrix)*threshold:
            break
    return 0,0
