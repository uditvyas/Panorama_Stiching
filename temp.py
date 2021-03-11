def drawMatches(img1, kp1, img2, kp2, matches, inliers=None):
    # Create a new output image that concatenates the two images together
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1, rows2]), cols1+cols2, 3), dtype='uint8')

    # Place the first image to the left
    out[:rows1, :cols1, :] = img1  # np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2, cols1:cols1+cols2, :] = img2  # np.dstack([img2, img2, img2])

    # For each pair of points we have between both images

    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns, y - rows
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        inlier = False

        if inliers is not None:
            for i in inliers:
                if i[0] == x1 and i[1] == y1 and i[2] == x2 and i[3] == y2:
                    inlier = True

        # Draw a small circle at both co-ordinates
        cv2.circle(out, (int(x1), int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1, int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points, draw inliers if we have them
        if inliers is not None and inlier:
            cv2.line(out, (int(x1), int(y1)),
                     (int(x2)+cols1, int(y2)), (0, 255, 0), 1)
        elif inliers is not None:
            cv2.line(out, (int(x1), int(y1)),
                     (int(x2)+cols1, int(y2)), (0, 0, 255), 1)

        if inliers is None:
            cv2.line(out, (int(x1), int(y1)),
                     (int(x2)+cols1, int(y2)), (255, 0, 0), 1)

    return out
