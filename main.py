# This file contains the driver code for Panorama Stitching
from helpers import *

image1 = cv2.imread("inputImages/I4/DSC02930.JPG")
image2 = cv2.imread("inputImages/I4/DSC02931.JPG")
image3 = cv2.imread("inputImages/I4/DSC02932.JPG")
image4 = cv2.imread("inputImages/I4/DSC02933.JPG")
image5 = cv2.imread("inputImages/I4/DSC02934.JPG")
# Assuming we take image 2 as the reference image, we transfxorm rest all images to Image 2's perpective

image1 = cv2.resize(image1, (800,600))
image2 = cv2.resize(image2, (800,600))
image3 = cv2.resize(image3, (800,600))
image4 = cv2.resize(image4, (800,600))
image5 = cv2.resize(image5, (800,600))

H12 = stich(image1, image2)
H32 = stich(image3, image2)
H43 = stich(image4, image3)
H54 = stich(image5, image4)
H42 = np.dot(H43, H32)
H52 = np.dot(H54, H42)

height = int(3*image1.shape[0])
width = 6*image1.shape[1]

x_offset = int(image1.shape[1])
y_offset = int(1.5*image1.shape[0])

output = setReference(image2, x_offset, y_offset, height, width)

mask, original, warped = mywarp(output, image1, x_offset, y_offset, H12)
output = blend(original, warped, mask, stichOnLeft=1)   

mask, original, warped =  mywarp(output, image3, x_offset, y_offset, H32)
output = blend(original, warped, mask, stichOnLeft=0)

mask, original, warped =  mywarp_far(output, image4, x_offset, y_offset, H42)
output = blend(original, warped, mask, stichOnLeft=0)

mask, original, warped =  mywarp_far(output, image5, x_offset, y_offset, H52)
output = blend(original, warped, mask, stichOnLeft=0)

cv2.imwrite("panorama.png", output)
