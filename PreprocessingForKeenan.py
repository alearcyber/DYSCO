import numpy as np
import cv2

#color enums
red = (0, 0, 255)
green = (0, 255, 0)


#read in image and find contours
im = cv2.imread('/Users/aidan/Desktop/Snapchat-148520670.jpg')
assert im is not None, "file could not be read, check with os.path.exists()"
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



# draw the contours found so they can be visualized
n = 5 # number of contours
cnt = sorted(contours, key=cv2.contourArea, reverse=True)
cnt = cnt[:n] # 5 biggest contours (by area)
contour_selections = []
for i in range(5): # i is contour to highlight
    out = np.copy(im)
    for j in range(5): #j is contour currently being drawn
        if j == i:
            out = cv2.drawContours(out, cnt, j, green, 3)
        else:
            out = cv2.drawContours(out, cnt, j, red, 3)
    contour_selections.append(out)



#begin displaying contours for the user
image_selected = 0
cv2.imshow('Press Enter To Select Contour', contour_selections[image_selected])
while True:
    key = cv2.waitKey(0) & 0xFF  # Wait for a key press and get the ASCII code

    if key == ord('n') or key == 3:  # If 'n' or right arrow is pressed, go to the next image
        image_selected = (image_selected + 1) % len(contour_selections)
        cv2.imshow('Press Enter To Select Contour', contour_selections[image_selected])

    elif key == ord('b') or key == 2:  # If 'b' or left arrow is pressed, go to the previous image
        image_selected = (image_selected - 1) % len(contour_selections)
        cv2.imshow('Press Enter To Select Contour', contour_selections[image_selected])

    else:
        break  # Exit on any other key press
cv2.destroyAllWindows()



#approximate a quadrilateral from contour (currently implemented using convex hull, then Douglas-Peucker algo to 4 sides)
contour = cnt[image_selected]
hull = cv2.convexHull(contour)
out = np.copy(im)
out = cv2.drawContours(out, [hull], 0, green, 3)
cv2.imshow('Quadrilateral Approximation of Contour', out)
cv2.waitKey()

#Douglas Peucker
desired_number_of_sides = 4
max_attempts = 10
gamma = 0.5
success = False
for _ in range(max_attempts):
    epsilon = gamma * cv2.arcLength(contour, True)  # example gamma value gives waws 0.1
    approx = cv2.approxPolyDP(contour, epsilon, True)
    number_of_vertices = approx.shape[0]
    if number_of_vertices == desired_number_of_sides:
        success = True
        break
    gamma -= 0.05

#check if approximation worked
assert success, "ERROR, could not approximate quadrilateral"



#now perform the perspective (homographic) transformation
print(approx)
vertices = []
for i in range(4):
    vertex = approx[i].reshape(2)
    vertices.append(vertex)


#midpoints
midpoints = np.array([
        (vertices[0] + vertices[1])//2,
        (vertices[1] + vertices[2])//2,
        (vertices[2] + vertices[3])//2,
        (vertices[3] + vertices[0])//2,
    ])
x_hi = np.amax(midpoints[:, 0])
x_lo = np.amin(midpoints[:, 0])
y_hi = np.amax(midpoints[:, 1])
y_lo = np.amin(midpoints[:, 1])

#clockwise ordering
new_vertices = np.array([
    [x_lo, y_lo],
    [x_hi, y_lo],
    [x_hi, y_hi],
    [x_lo, y_hi]
])
out = np.copy(im)
out = cv2.drawContours(out, [new_vertices], 0, green, 3)
cv2.imshow('Perspective goal shape with perspective alignment.', out)
cv2.waitKey()


#make match vertices to prepare for transformation
vertices = np.array(vertices)
print('---------')
print('old:', vertices)
print('new:', new_vertices)

#first vertex
ordered_new_vertices = np.copy(new_vertices)
for i in range(4):
    distance = np.array([np.linalg.norm(vertices[i] - new_vertices[j]) for j in range(4)]).astype(np.int32)
    ordered_index, = np.where(distance == np.amin(distance))
    ordered_new_vertices[ordered_index] = new_vertices[i]

new_vertices = np.copy(ordered_new_vertices)


#Perform perspective(homographic) transformation
homographic_matrix = cv2.getPerspectiveTransform(vertices.astype(np.float32), new_vertices.astype(np.float32))
print(homographic_matrix)
result = cv2.warpPerspective(im, homographic_matrix, (im.shape[1], im.shape[0]))
cv2.imshow('Perspective corrected', result)
cv2.waitKey()

#crop image to size
cropped = result[y_lo: y_hi+1, x_lo: x_hi+1]
cv2.imshow('Cropped', cropped)
cv2.waitKey()


#threshold


#Clahe
#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#clahe = cv2.createCLAHE()
#normalized = clahe.apply(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY))
#cv2.imshow('normalized', normalized)
#cv2.waitKey()

print(cropped.dtype)
print(cropped.shape)
cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
#unsharp mask
blur = cv2.GaussianBlur(cropped, (0, 0), 2.0)
cropped = cv2.addWeighted(cropped, 2.0, blur, -1.0, 0)
cv2.imshow('Image Sharpening (unsharp mask)', cropped)
cv2.waitKey()
#cropped = cv2.medianBlur(cropped, 3)
#cropped = cv2.GaussianBlur(cropped, (3,3), 0)
#th = cv2.adaptiveThreshold(cropped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)
_, th = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('threshold', th)
cv2.waitKey()



"""
for i in range(max_attempts):
    epsilon = gamma * cv2.arcLength(contour, True)  # example gamma value gives waws 0.1
    approx = cv2.approxPolyDP(contour, epsilon, True)
    number_of_vertices = approx.shape[0]
    print(f'gamma:{gamma}, vertices:{number_of_vertices}')
    gamma -= 0.05
"""








"""
print(len(cnt))
print(type(cnt[0]))
print(cv2.contourArea(cnt[0]))




#draw contours
cnt = cnt[:5]
red = (0, 0, 255)
green = (0, 255, 0)
out = cv2.drawContours(im, cnt, -1, green, 3)
#out = cv2.drawContours(im, cnt, 0, (0,255,0), 3)
cv2.imshow('', out)
cv2.waitKey()
"""
