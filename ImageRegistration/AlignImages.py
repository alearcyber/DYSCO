
import cv2
import numpy as np
import matplotlib.pyplot as plt


img = [  # images
    cv2.imread("images/fail1.jpg"),
    cv2.imread("images/fail2.jpg"),
    cv2.imread("images/fail3.jpg"),
    cv2.imread("images/fail4.jpg"),
    cv2.imread("images/fail6.jpg"),
]

dash = cv2.imread("/Users/aidanlear/Desktop/Research2023/TESTIMAGES/dash1.jpg")
dash_s = cv2.imread("images/dash4.jpg") # screenshot of the dashboard. Expected display output




#shows an image
def show(image, title=None):
    plt.imshow(image)
    if not (title is None):
        plt.title(title)
    plt.show()


#does the morphology thing
def morph(image):
    im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(im, 127, 255, cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    m = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return m



#extend image bounds with blank pixels so they match.
def extend_image(a, b):
    #check if the image is grayscale
    if len(a.shape) == 2:
        return _extend_image_grayscale(a, b)


    #extend height
    dx = a.shape[0] - b.shape[0]
    if dx > 0: #a is bigger, extend b
        new_pixels = np.zeros((abs(dx), b.shape[1], 3), dtype='uint8')
        b = np.concatenate((b, new_pixels), axis=0)
    elif dx < 0: #b is bigger, extend a
        new_pixels = np.zeros((abs(dx), a.shape[1], 3), dtype='uint8')
        a = np.concatenate((a, new_pixels), axis=0)

    #extend width
    dy = a.shape[1] - b.shape[1]
    if dy > 0:  # a is bigger, extend b
        new_pixels = np.zeros((b.shape[0], abs(dy), 3), dtype='uint8')
        b = np.concatenate((b, new_pixels), axis=1)
    elif dy < 0:  # b is bigger, extend a
        new_pixels = np.zeros((a.shape[0], abs(dy), 3), dtype='uint8')
        a = np.concatenate((a, new_pixels), axis=1)

    #return output
    return a, b


#helper function to extend grayscale images
def _extend_image_grayscale(a, b):
    # extend height
    dx = a.shape[0] - b.shape[0]
    if dx > 0:  # a is bigger, extend b
        new_pixels = np.zeros((abs(dx), b.shape[1]), dtype='uint8')
        b = np.concatenate((b, new_pixels), axis=0)
    elif dx < 0:  # b is bigger, extend a
        new_pixels = np.zeros((abs(dx), a.shape[1]), dtype='uint8')
        a = np.concatenate((a, new_pixels), axis=0)

    # extend width
    dy = a.shape[1] - b.shape[1]
    if dy > 0:  # a is bigger, extend b
        new_pixels = np.zeros((b.shape[0], abs(dy)), dtype='uint8')
        b = np.concatenate((b, new_pixels), axis=1)
    elif dy < 0:  # b is bigger, extend a
        new_pixels = np.zeros((a.shape[0], abs(dy)), dtype='uint8')
        a = np.concatenate((a, new_pixels), axis=1)

    # return output
    return a, b




#Aligns query image with the train image based on their features and estimating the affine transformation.
#Returns a copy of the query image and the train image. This is because both images may be given padding so they
# have the same dimensions.
#Note: Does not currently consider the accuracy/strength of the matches.
def align_images(query, train):
    # extend bounds of images
    query, train = extend_image(query, train)

    # keypoints and matching
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(query, None)
    kp2, des2 = sift.detectAndCompute(train, None)
    bf = cv2.BFMatcher(cv2.DIST_L2, crossCheck=True)
    matches = bf.match(des1, des2)  # first is query, second is train
    matches = sorted(matches, key=lambda x: x.distance)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find the transformation between points, standard RANSAC
    transformation_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    #Warp the image
    query = cv2.warpPerspective(query, transformation_matrix, (query.shape[1], query.shape[0]))

    # return the aligned images
    return query, train


#Division Normalization
def div_norm(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) #grayscale
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=33, sigmaY=33) # blur
    divide = cv2.divide(gray, blur, scale=255)
    #_, thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #_, thresh = cv2.threshold(divide, 127, 255, cv2.THRESH_BINARY)
    _, thresh = cv2.threshold(divide, 127, 255, cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return morph





def test_extend_image():
    #cv2.imshow('before a', cv2.imread("/Users/aidanlear/Desktop/trash1.png"))
    #cv2.imshow('before b', cv2.imread("/Users/aidanlear/Desktop/trash2.png"))
    #cv2.waitKey()
    a, b = extend_image(cv2.imread("/Users/aidanlear/Desktop/trash1.png"), cv2.imread("/Users/aidanlear/Desktop/trash2.png"))

    cv2.imshow('a', a)
    cv2.imshow('b', b)
    cv2.waitKey()





#example code from online, seems to work as advertised.
def ecc_example(im1, im2):
    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Find size of image1
    sz = im1.shape

    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 5000

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria)

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective(im2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    # Show final results
    cv2.imshow("Image 1", im1)
    cv2.imshow("Image 2", im2)
    cv2.imshow("Aligned Image 2", im2_aligned)
    cv2.waitKey(0)



def ecc_combine(im1, im2):
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Find size of image1
    sz = im1.shape

    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 5000

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria)

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective(im2, warp_matrix, (sz[1], sz[0]),
                                          flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    # Show final results
    #cv2.imshow("Image 1", im1)
    #cv2.imshow("Image 2", im2)
    #cv2.imshow("Aligned Image 2", im2_aligned)
    #cv2.waitKey(0)

    out = np.add(im1, im2) // 2
    cv2.imshow('combined',)



#tests the estimateAffine2D() method on opencv
def estimate_affine():

    #extend bounds of images
    img[0], img[1] = extend_image(img[0], img[1])
    print('shape1:', img[0].shape)
    print('shape2:', img[1].shape)

    #keypoints and matching
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img[0], None)
    kp2, des2 = sift.detectAndCompute(img[1], None)
    bf = cv2.BFMatcher(cv2.DIST_L2, crossCheck=True)
    matches = bf.match(des1, des2)  # first is query, second is train
    matches = sorted(matches, key=lambda x: x.distance)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find the transformation between points, standard RANSAC
    transformation_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Compute a rigid transformation (without depth, only scale + rotation + translation) and RANSAC
    transformation_rigid_matrix, rigid_mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)

    print('transformation matrix:', transformation_matrix)

    print('rigid:', transformation_rigid_matrix)

    out = cv2.warpPerspective(img[0], transformation_matrix, (img[0].shape[1], img[0].shape[0]))
    #out = img[0] * transformation_matrix
    #cv2.imshow('original', img[0])
    #cv2.imshow('train', img[1])

    out = np.add(out, img[1]) // 2
    cv2.imshow('out', out)
    cv2.waitKey()



def test_align_images():
    a, b = align_images(img[0], img[3])
    out = np.add(a, b) // 2
    cv2.imshow('out', out)
    cv2.waitKey()




#testing "frame differencing" to compare different images
#Logic here is to threshold the difference between the aligned images
#This specific test here is just simple image differencing
def frame_diff_test():
    #align the images
    a, b = align_images(img[0], img[3])

    #difference the images
    out = abs(a - b)

    #convert to grayscale
    out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

    # threshold
    _, out = cv2.threshold(out, 127, 255, cv2.THRESH_BINARY)

    #display the result
    cv2.imshow('out', out)
    cv2.waitKey()


#Division Normalization
def div_norm_test(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) #grayscale
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=33, sigmaY=33) # blur
    divide = cv2.divide(gray, blur, scale=255)
    #_, thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thresh = cv2.threshold(divide, 127, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("blur", blur)
    cv2.imshow("divide", divide)
    cv2.imshow("thresh", thresh)
    cv2.imshow("morph", morph)
    cv2.waitKey()


#Differencing images after processing division normalization
def test2():
    cv2.imshow('query', img[0])
    cv2.imshow('train', dash_s)
    a, b = align_images(img[0], dash_s)

    a = div_norm(a)
    b = div_norm(b)
    out = abs(a - b)
    cv2.imshow('out', out)
    cv2.waitKey()


#compare the division normalization difference for expected and capture where
# it is known that there is no obstruction
def baseline_difference_test():
    #checking grayscale information
    gray = cv2.cvtColor(dash, cv2.COLOR_BGR2GRAY)
    print('gray dimensions:', gray.shape)
    print('gray dtype:', gray.dtype)


    #get baseline difference
    a, b = align_images(dash, dash_s)
    a = div_norm(a)
    b = div_norm(b)
    baseline = abs(a - b)

    #difference with obstruction
    a, b = align_images(img[4], dash_s)
    a = div_norm(a)
    b = div_norm(b)
    out = abs(a - b)


    #whoa thats crazy
    out, baseline = align_images(out, baseline)


    double_out = abs(out - baseline)
    plt.imshow(double_out)
    plt.show()
   # cv2.imshow('double out', double_out)
    #cv2.waitKey()


#preprocessing pipeline
def preprocessing_pipeline(im):
    #division normalization
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # grayscale
    show(gray, 'original')
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=33, sigmaY=33)  # blur
    show(blur, 'blur')

    divide = cv2.divide(gray, blur, scale=255)
    show(divide, 'divide')
    # _, thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # _, thresh = cv2.threshold(divide, 127, 255, cv2.THRESH_BINARY)
    #_, thresh = cv2.threshold(divide, 127, 255, cv2.THRESH_OTSU)
    #thresh = cv2.adaptiveThreshold(divide, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    #thresh = cv2.adaptiveThreshold(divide, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.adaptiveThreshold(divide, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_OTSU, 11, 2)
    show(thresh, 'threshold')


    _, thresh2 = cv2.threshold(thresh, 240, 255, cv2.THRESH_OTSU)
    show(thresh2, 'second threshold')

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return morph


def test_morph():
    #align the images
    a, b = align_images(img[3], dash)
    #a, b = align_images(img[0], dash)
    show(a, 'aligned')
    show(b, 'aligned')

    #morphology process
    out = abs(morph(a) - morph(b))
    show(out, "difference")


    #draw contours
    contours, hierarchy = cv2.findContours(out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, 3, (0, 255, 0), 3)
    show(out, 'out')







if __name__ == '__main__':
    #baseline_difference_test()
    #preprocessing_pipeline(img[3])
    test_morph()










