# This file contains all the helper functions for Panorama Stitching

import cv2
import numpy as np
import random
from tqdm import tqdm

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
            valid_matches.append(k_match[0])
    return valid_matches


def correspondence_matrix(valid_matches, ref_keypoints, tar_keypoints):
    keypoints = []

    for match in valid_matches:
        (x0, y0) = ref_keypoints[match.queryIdx].pt
        (x1, y1) = tar_keypoints[match.trainIdx].pt
        keypoints.append([x0, y0, x1, y1])

    return keypoints


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


def findError(matrixRow, H):
    point1 = np.transpose(np.array([matrixRow[0], matrixRow[1], 1]))
    point2 = np.array([matrixRow[2], matrixRow[3], 1])
    estimate = np.dot(H, point1)
    estimate = estimate/estimate[0, 2]
    error = point2 - estimate
    return np.linalg.norm(error)


def get_homography(matrix, threshold, n_iter=1500):
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

        # if len(inliers) > len(matrix)*threshold:
        #     break
    return finalH  # inliers


def stich(reference, target):
    (reference_keypoints, reference_features) = extract_features_keypoints(reference)
    (target_keypoints, target_features) = extract_features_keypoints(target)
    all_matches = match_features(reference_features, target_features)
    good_matches = valid_matches(all_matches)
    keypoint_matrix = correspondence_matrix(
        good_matches, reference_keypoints, target_keypoints)
    threshold = 1

    H = get_homography(keypoint_matrix, threshold)
    return np.array(H)

def setReference(image, x_offset, y_offset, x, y):
    warped = np.zeros((x, y, 3))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            warped[i+x_offset][j+y_offset] = image[i][j]
    print("Reference Image Set!")
    return warped


def mywarp(output, target, x_offset, y_offset, H):
    out = output.copy()
    output_copy = np.zeros_like(output)
    for i in tqdm(range(target.shape[0])):
        for j in range(target.shape[1]):
            computed = H.dot(np.asarray([j, i, 1]).T)
            computed = computed/computed[2]
            x_c = int(computed[0])        
            y_c = int(computed[1])
            try:
                for i1 in range(-1, 2):
                    for i2 in range(-1, 2):
                        output_copy[y_c+x_offset+i1][x_c + y_offset+i2] = target[i, j]
                        output[y_c+x_offset+i1][x_c + y_offset+i2] = target[i, j]
            except:
                continue
    mask = out*output_copy
    return mask,out, output_copy

def mywarp_far(output, target, x_offset, y_offset, H):
    out = output.copy()
    output_copy = np.zeros_like(output)
    for i in tqdm(range(target.shape[0])):
        for j in range(target.shape[1]):
            computed = H.dot(np.asarray([j, i, 1]).T)
            computed = computed/computed[2]
            x_c = int(computed[0])     
            y_c = int(computed[1])
            try:
                for i1 in range(-2, 3):
                    for i2 in range(-2, 3):
                        output_copy[y_c+x_offset+i1][x_c + y_offset+i2] = target[i, j]
                        output[y_c+x_offset+i1][x_c + y_offset+i2] = target[i, j]
            except:
                continue
    mask = out*output_copy
    return mask,out, output_copy

def blend(output, B, mask, stichOnLeft, levels = 6):
    mask = np.where(mask>0, 0, 1)
    # A = output*mask
    A = output
    xmin, xmax = np.min(np.where(mask == 0)[1]), np.max(np.where(mask == 0)[1])
    xmid = (xmin + xmax)//2
    new_mask = np.zeros_like(output)
    if stichOnLeft:
        new_mask[:, xmid:, :] = 1
    else:
        new_mask[:, :xmid, :] = 1

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
    
    # lpB = [gpB[levels-1]]
    # for i in range(levels-1,0,-1):
    #     GE = cv2.pyrUp(gpB[i], dstsize=(gpB[i-1].shape[1], gpB[i-1].shape[0]))
    #     L = cv2.subtract(gpB[i-1],GE)
    #     lpB.append(L)

    LS = []
    for la,lb,gp in zip(lpA,lpB, gp_new[::-1]):
        # print(gp.shape)
        # print(la.shape)
        # rows,cols,dpt = la.shape
        ls = gp*la + (1-gp)*lb
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in range(1,levels):
        ls_ = cv2.pyrUp(ls_, dstsize=(LS[i].shape[1], LS[i].shape[0]))
        ls_ = cv2.add(ls_, LS[i])
    return ls_