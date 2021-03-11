# This file contains the driver code for Panorama Stitching
from helpers import *

image1 = cv2.imread("inputImages/I2/2_2.JPG")
image2 = cv2.imread("inputImages/I2/2_3.JPG")
image3 = cv2.imread("inputImages/I2/2_4.JPG")
image4 = cv2.imread("inputImages/I2/2_5.JPG")

# Assuming we take image 2 as the reference image, we transfxorm rest all images to Image 2's perpective

H12 = stich(image1, image2)
H32 = stich(image3, image2)
H43 = stich(image4, image3)
H42 = np.dot(H43, H32)

val = image2.shape[1] + image1.shape[1]

output = warp(image1, image2, image3, image4, int(0.15*image1.shape[0]),
              image1.shape[1], int(1.3*image1.shape[0]), 4*image1.shape[1], H12, H32, H42)
cv2.imwrite("panorama.png", output)
