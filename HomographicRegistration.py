import cv2

image = cv2.imread("/Users/aidanlear/Desktop/teapot-static.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

selected_contour = 0
key = None
frame = cv2.drawContours(image, contours[selected_contour], -1, (0, 255, 0), 5)
#loop to select contour
while True:
	if key == ord('q'):
		break
	elif key == ord('b'):
		print("pressed b")
	cv2.imshow("frame", frame)
	key = cv2.waitKey(10) 
print("done")

	
	

drawn = cv2.drawContours(image, contours[:10], -1, (0,255,0), 5)
cv2.imshow("", drawn)

cv2.waitKey()

