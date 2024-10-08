import numpy as np
import cv2
import matplotlib.pyplot as plt



#grab alpha image
image = cv2.imread("/Users/aidan/Desktop/BrandNewObstructions/CroppedObstructions/obstruction14.png", cv2.IMREAD_UNCHANGED)
alpha = image[:, :, 3]


#erode
d = 13
kernel = np.ones((d, d), np.uint8)
#erosion = cv2.erode(alpha, kernel, iterations=1)

#closing
closing = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)



#erode the closing

#print(np.count_nonzero(closing > 0 & closing < 255))

cv2.imshow("alpha", alpha)
cv2.imshow("closing", closing)
cv2.imshow("opening", cv2.morphologyEx(alpha, cv2.MORPH_DILATE, np.ones((17, 17), np.uint8)))
cv2.waitKey()
