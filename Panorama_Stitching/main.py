# This file contains the driver code for Panorama Stitching

from helpers import *

#### Part 1: The following code lines take input of the images in sets. 

# Uncomment one set at a time to take the right input
# Sets with very large images have been resized just below the input.
# Uncomment the resizing code if it follows the input code

# # SET 1
image1 = cv2.imread("inputImages/I1/STC_0033.JPG")
image2 = cv2.imread("inputImages/I1/STD_0034.JPG")
image3 = cv2.imread("inputImages/I1/STE_0035.JPG")
image4 = cv2.imread("inputImages/I1/STF_0036.JPG")
out = 1

# RESIZE Set 1
image1 = cv2.resize(image1, (800,600))
image2 = cv2.resize(image2, (800,600))
image3 = cv2.resize(image3, (800,600))
image4 = cv2.resize(image4, (800,600))

# SET 2
# image1 = cv2.imread("inputImages/I2/2_1.JPG")
# image2 = cv2.imread("inputImages/I2/2_2.JPG")
# image3 = cv2.imread("inputImages/I2/2_3.JPG")
# image4 = cv2.imread("inputImages/I2/2_4.JPG")
# out = 2

# SET 3
# image2 = cv2.imread("inputImages/I3/3_3.JPG")
# image3 = cv2.imread("inputImages/I3/3_4.JPG")
# image4 = cv2.imread("inputImages/I3/3_5.JPG")
# out = 3

# SET 4
# image1 = cv2.imread("inputImages/I4/DSC02930.JPG")
# image2 = cv2.imread("inputImages/I4/DSC02931.JPG")
# image3 = cv2.imread("inputImages/I4/DSC02932.JPG")
# image4 = cv2.imread("inputImages/I4/DSC02933.JPG")
# out = 4

# # RESIZE Set 4
# image1 = cv2.resize(image1, (800,600))
# image2 = cv2.resize(image2, (800,600))
# image3 = cv2.resize(image3, (800,600))
# image4 = cv2.resize(image4, (800,600))


# SET 5
# image1 = cv2.imread("inputImages/I5/DSC03002.JPG")
# image2 = cv2.imread("inputImages/I5/DSC03003.JPG")
# image3 = cv2.imread("inputImages/I5/DSC03004.JPG")
# image4 = cv2.imread("inputImages/I5/DSC03005.JPG")
# out = 5

# # RESIZE Set 5
# image1 = cv2.resize(image1, (800,600))
# image2 = cv2.resize(image2, (800,600))
# image3 = cv2.resize(image3, (800,600))
# image4 = cv2.resize(image4, (800,600))

# SET 6
# image1 = cv2.imread("inputImages/I6/1_1.JPG")
# image2 = cv2.imread("inputImages/I6/1_2.JPG")
# image3 = cv2.imread("inputImages/I6/1_3.JPG")
# image4 = cv2.imread("inputImages/I6/1_4.JPG")
# out = 6


#### Part 2: Preparing the Output Image

height = int(3*image2.shape[0])
width = 5*image2.shape[1]
x_offset = int(image2.shape[1])
y_offset = int(image2.shape[0])
output = setReference(image2, x_offset, y_offset, height, width)


#### Part 3: Calculating the Homography Matrix, Warping and Blending - one image at a time.

# Note: For SET 3, Comment the subsequent three lines. Otherwise, it will throw error, since Image1 in not defined
# The reason is explained in the report submitted
H12 = stich(image1, image2)
mask, original, warped = mywarp(output, image1, x_offset, y_offset, H12)
output = blend(original, warped, mask, stichOnLeft=1)

###

H32 = stich(image3, image2)
mask, original, warped =  mywarp(output, image3, x_offset, y_offset, H32)
output = blend(original, warped, mask, stichOnLeft=0)

###

# EXCEPTION WARNING: FOR SET 6
# Exchange the mywarp_far function with mywarp function. Change the comments appropriately

# For SET 1 to 5
H43 = stich(image4, image3)
H42 = np.dot(H43, H32)
mask, original, warped =  mywarp_far(output, image4, x_offset, y_offset, H42)
output = blend(original, warped, mask, stichOnLeft=0)

# FOR SET - 6
# H42 = stich(image4, image2)
# mask, original, warped =  mywarp(output, image4, x_offset, y_offset, H42)

###

cv2.imwrite("results/panorama_"+str(out)+".png", output)