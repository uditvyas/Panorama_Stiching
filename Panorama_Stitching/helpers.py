# This file contains all the helper functions for Panorama Stitching

import cv2
import numpy as np
import random
from tqdm import tqdm

# Function to extract SIFT like keypoints and corresponding descriptors from the image
def extract_features_keypoints(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    (Keypoints, features) = orb.detectAndCompute(image, None)
    return (Keypoints, features)

# Function to match the Features in two images using Brute force, KNN matcher (default K = 2)
def match_features(features1, features2, K=2):
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(features1, features2, K)
    return matches


# Function to find high confidence matches using the Lowe's ratio rule (default ratio test = 0.8)
def valid_matches(matches, lowe_ratio=0.8):
    valid_matches = []

    for k_match in matches:
        if len(k_match) == 2 and k_match[0].distance < k_match[1].distance * lowe_ratio:
            valid_matches.append(k_match[0])
    return valid_matches


# Function to prepare the keypoints based on the matches, to feed as an input to the RANSAC Algorithm
def correspondence_matrix(valid_matches, ref_keypoints, tar_keypoints):
    keypoints = []

    for match in valid_matches:
        (x0, y0) = ref_keypoints[match.queryIdx].pt
        (x1, y1) = tar_keypoints[match.trainIdx].pt
        keypoints.append([x0, y0, x1, y1])

    return keypoints

# Function to calculate a homography matrix from 4 samples provided, using SVD
def calcH(sample):
    A = []
    for x, y, xx, yy in sample:

        first = [x, y, 1, 0, 0, 0, -xx*x, -xx*y, -xx]
        second = [0, 0, 0, x, y, 1, -yy*x, -yy*y, -yy]

        A.append(first)
        A.append(second)

    A = np.matrix(A)
    U, S, V = np.linalg.svd(A)

    H = np.reshape(V[8], (3, 3))
    # Normalising
    H = (1/H[2, 2])*H
    return H


# Function to find error, by finding the difference between the projected value and the actual value
def findError(matrixRow, H):
    point1 = np.transpose(np.array([matrixRow[0], matrixRow[1], 1]))
    point2 = np.array([matrixRow[2], matrixRow[3], 1])
    estimate = np.dot(H, point1)
    estimate = estimate/estimate[0, 2]
    error = point2 - estimate
    return np.linalg.norm(error)

# Function implementing the RANSAC Algorithm ()
def get_homography(matrix, n_iter=1500):
    inliers = []
    n = len(matrix)
    finalH = None
    for i in tqdm(range(n_iter)):
        indices = random.sample(range(1, n), 4)
        random_sample = [matrix[i] for i in indices]

        H = calcH(random_sample)
        iteration_inliers = []

        for i in range(n):
            error = findError(matrix[i], H)
            if error < 2:
                iteration_inliers.append(matrix[i])

        if len(iteration_inliers) > len(inliers):
            inliers = iteration_inliers
            finalH = H

    return finalH

# Wrapper helper Function to find the Homography matrix
def stich(reference, target):
    (reference_keypoints, reference_features) = extract_features_keypoints(reference)
    (target_keypoints, target_features) = extract_features_keypoints(target)
    all_matches = match_features(reference_features, target_features)
    good_matches = valid_matches(all_matches)
    keypoint_matrix = correspondence_matrix(good_matches, reference_keypoints, target_keypoints)

    H = get_homography(keypoint_matrix)
    return np.array(H)

# Function to create an output image, and fix the reference image in the output image.
def setReference(image, x_offset, y_offset, x, y):
    warped = np.zeros((x, y, 3))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            warped[i+y_offset][j+x_offset] = image[i][j]
    print("Reference Image Set!")
    return warped

# Function to warp an image, and add it to the output image generated using the previous function
# This function also generates a mask of common values between the stitched image so far, and the new warped image
# The function returns the original image, the separate warped image, and as well as the mask as the output
def mywarp(output, target, x_offset, y_offset, H):
    out = output.copy()
    output_copy = np.zeros_like(output)
    h = target.shape[0]
    w = target.shape[1]
    
    # Finding the bounding rectange in the transformed plane.
    corners = [[0,0,1],[h-1,0,1],[0,w-1,1],[h-1,w-1,1]]
    transform_corners = np.array([H.dot(np.array(corner).T) for corner in corners])
    transform_corners = [corner/corner[2] for corner in transform_corners]
    xs = [corner[1] for corner in transform_corners]
    ys = [corner[0] for corner in transform_corners]
    xmin, xmax = int(np.min(xs)), int(np.max(xs))+1
    xmin = max(xmin, -x_offset)
    xmax = min(xmax + x_offset, output_copy.shape[1])
    ymin, ymax = int(np.min(ys)), int(np.max(ys))+1
    ymin = min(ymin, -y_offset)
    ymax = min(ymax + y_offset, output_copy.shape[0])

    invH = np.linalg.inv(H)
    
    # Reverse mapping the points in the transformed plane to check if they lie in the original images
    for j in tqdm(range(ymin,ymax)):
        for i in range(xmin,xmax):
            point = np.array([i,j,1]).T
            inverse = invH.dot(point)
            inverse = inverse/inverse[2]
            x = int(inverse[0])
            y = int(inverse[1])

            if x in range(0,w) and y in range(0,h):
                try:
                    output_copy[j + y_offset][i + x_offset] = target[y][x]
                    output[j + y_offset][i + x_offset] = target[y][x]
                except:
                    continue
    mask = out*output_copy
    return mask, out, output_copy

# This function is used to blend the newly warped image and the existing stiched image
# It makes use of the Image pyraminds to estimate Gaussian and Laplacian pyramids of the two images
# and makes used the mask generated in the previous step, to reconstruct the images in order to blend them
def blend(A, B, mask, stichOnLeft, levels = 6):
    mask = np.where(mask>0, 0, 1)
    xmin, xmax = np.min(np.where(mask == 0)[1]), np.max(np.where(mask == 0)[1])
    xmid = (xmin + xmax)//2
    new_mask = np.zeros_like(A)
    
    # This is conditional to the stitch being on the right side, or the left side of the reference image.
    if stichOnLeft:
        new_mask[:, xmid:, :] = 1
    else:
        new_mask[:, :xmid, :] = 1

    # generate Gaussian pyramid for A
    G = A.copy()
    gpA = [G]
    for i in range(levels):
        G = cv2.pyrDown(G)
        gpA.append(G)

    # generate Gaussian pyramid for B
    G = B.copy()
    gpB = [G]
    for i in range(levels):
        G = cv2.pyrDown(G)
        gpB.append(G)

    # generate Gaussian pyramid for the mask
    G = new_mask.copy()
    gp_new = [G]
    for i in range(1,levels):
        G = cv2.pyrDown(G)
        gp_new.append(G)

    # generate Laplacian Pyramid for A
    lpA = [gpA[levels-1]]
    for i in range(levels-1,0,-1):
        GE = cv2.pyrUp(gpA[i], dstsize=(gpA[i-1].shape[1], gpA[i-1].shape[0]))
        L = cv2.subtract(gpA[i-1],GE)
        lpA.append(L)

    # generate Laplacian Pyramid for B
    lpB = [gpB[levels-1]]
    for i in range(levels-1,0,-1):
        GE = cv2.pyrUp(gpB[i], dstsize=(gpB[i-1].shape[1], gpB[i-1].shape[0]))
        L = cv2.subtract(gpB[i-1],GE)
        lpB.append(L)

    LS = []
    for la,lb,gp in zip(lpA,lpB, gp_new[::-1]):
        ls = gp*la + (1-gp)*lb
        LS.append(ls)

    # now reconstruct
    result = LS[0]
    for i in range(1,levels):
        result = cv2.pyrUp(result, dstsize=(LS[i].shape[1], LS[i].shape[0]))
        result = cv2.add(result, LS[i])
    return result