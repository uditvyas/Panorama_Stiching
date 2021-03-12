# This file contains the driver code for Panorama Stitching
from helpers import *

image1 = cv2.imread("inputImages/I2/2_1.JPG")
image2 = cv2.imread("inputImages/I2/2_2.JPG")
image3 = cv2.imread("inputImages/I2/2_3.JPG")
image4 = cv2.imread("inputImages/I2/2_4.JPG")
# Assuming we take image 2 as the reference image, we transfxorm rest all images to Image 2's perpective

H12 = stich(image1, image2)
H32 = stich(image3, image2)
H43 = stich(image4, image3)
H42 = np.dot(H43, H32)

x_offset = int(0.35*image1.shape[0])
y_offset = image1.shape[1]

height = int(1.8*image1.shape[0])
width = 4*image1.shape[1]

output = setReference(image2, x_offset, y_offset, int(1.8*image1.shape[0]), 4*image1.shape[1])
mywarp(output, image1, x_offset, y_offset, H12)
mywarp(output, image3, x_offset, y_offset, H32)
mywarp_far(output, image4, x_offset, y_offset, H42)
cv2.imwrite("panorama.png", output)
