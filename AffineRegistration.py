"""
This file will contain all the routines for affine registration.
"""
import Color
import cv2
import numpy as np
import SimpleITK as sitk




def find_homography(moving, static, r=0.75):
    """
    :param moving: moving image
    :param static: static image
    :return: Matrix M that transforms
    Uses SIFT keypoints
    ratio test is
    """
    # finding SIFT keypoints
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(moving, None)
    keypoints2, descriptors2 = sift.detectAndCompute(static, None)

    print(len(descriptors1))
    print(len(descriptors2))

    # match keypoints
    index_params = dict(algorithm=(FLANN_INDEX_KDTREE := 1), trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Lowe's ratio test to remove bad matches
    good = []
    for m, n in matches:
        if m.distance < (r * n.distance):
            good.append(m)

    # Computer homography matrix from keypoints
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) # M is transformation matrix

    return M

def test_gftt():
    # Parameters for goodFeaturesToTrack
    maxCorners = 25  # Maximum number of corners to return
    qualityLevel = 0.01  # Minimum accepted quality of corners
    minDistance = 10  # Minimum distance between corners

    static_blue = cv2.imread("Data/Displays/box-blue.png", cv2.IMREAD_GRAYSCALE)

    features = cv2.goodFeaturesToTrack(static_blue, maxCorners, qualityLevel, minDistance)

    print(len(features))


    for f in features:
        print(f)
    print(static_blue.shape)

    # Display the image with corners
    """"
    #draw the features on the pictures
    if features is not None:
        corners = np.int8(features)
        for i in corners:
            x, y = i.ravel()
            cv2.circle(static_blue, (x, y), 3, (0, 0, 255), -1)
            
    cv2.imshow('Features Detected', static_blue)
    cv2.waitKey()
    cv2.destroyAllWindows()
    """

def sitk_to_homogeneous(transform: sitk.AffineTransform):
    A = np.array(transform.GetMatrix()).reshape(2, 2)
    t = np.array(transform.GetTranslation())
    c = np.array(transform.GetCenter())

    # Step 1: Translate to origin
    T1 = np.eye(3)
    T1[:2, 2] = -c

    # Step 2: Affine matrix
    T2 = np.eye(3)
    T2[:2, :2] = A

    # Step 3: Translate back
    T3 = np.eye(3)
    T3[:2, 2] = c

    # Step 4: Final translation
    T4 = np.eye(3)
    T4[:2, 2] = t

    # Compose the total transform
    H = T4 @ T3 @ T2 @ T1
    return H

def test_sitk():

    moving = cv2.imread("Data/Mar14Tests/0021.jpg", cv2.IMREAD_GRAYSCALE)
    static = cv2.imread("Data/Displays/box-blue.png", cv2.IMREAD_GRAYSCALE)

    moving = cv2.resize(moving, (1600, 900), interpolation=cv2.INTER_AREA)
    cv2.imshow('moving', moving)
    cv2.waitKey()

    moving, static = moving.astype(np.float32), static.astype(np.float32)




    moving = sitk.GetImageFromArray(moving, isVector=False)
    static = sitk.GetImageFromArray(static, isVector=False)


    #static = sitk.ReadImage("Data/Displays/box-blue.png", sitk.sitkFloat32)
    #moving = sitk.ReadImage("Data/Mar14Tests/0021.jpg", sitk.sitkFloat32)


    ##########################################
    # Transformation code
    ##########################################
    # Initialize registration
    registration = sitk.ImageRegistrationMethod()

    # Similarity metric
    #registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricAsMeanSquares()
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.01)

    # Interpolator
    registration.SetInterpolator(sitk.sitkLinear)

    # Optimizer
    registration.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=200,
                                               convergenceMinimumValue=1e-6,
                                               convergenceWindowSize=10)
    registration.SetOptimizerScalesFromPhysicalShift()

    # Transformation
    initial_transform = sitk.CenteredTransformInitializer(
        static, moving, sitk.AffineTransform(static.GetDimension()), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration.SetInitialTransform(initial_transform, inPlace=False)

    # Execute registration; calculate transformation
    final_transform: sitk.CompositeTransform = registration.Execute(static, moving)

    #print(final_transform)
    affine_transform: sitk.AffineTransform = final_transform.GetNthTransform(0)
    print(affine_transform)
    t11, t12, t21, t22 = affine_transform.GetInverse().GetMatrix()
    t13, t23 = affine_transform.GetTranslation()
    print(affine_transform.GetParameters())

    #move to origin
    T1 = np.array([
        [52]
    ])


    M = np.array([[t11, t12, t13],
                  [t21, t22, t23]])

    M = np.array([[t11, t12, t13],
                  [t21, t22, t23],
                  [0, 0, 1]])

    print(M)

    print(np.linalg.inv(M))
    M = M[0:2, :]
    print(M)





    # apply transformation
    registered = sitk.Resample(moving, static, final_transform, sitk.sitkLinear, 0.0, moving.GetPixelID())


    # display
    registered_np = sitk.GetArrayFromImage(registered) # Convert to NumPy array

    # normalize to 0–255 and convert to uint8 for OpenCV display
    registered_np_uint8 = cv2.normalize(registered_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Display using OpenCV
    cv2.imshow("Registered", registered_np_uint8)
    cv2.waitKey(0)

    blue = cv2.imread("Data/Mar14Tests/0021.jpg")
    blue = cv2.resize(blue, (1600, 900), interpolation=cv2.INTER_AREA)
    blue = cv2.warpAffine(blue, M, (1600, 900))
    cv2.imshow("Registered", blue)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





def test_images():
    """helper routine to grab the test images"""
    blue = cv2.imread("Data/Mar14Tests/0021.jpg")
    red = cv2.imread("Data/Mar14Tests/0019.jpg")

    static_blue = cv2.imread("Data/Displays/box-blue.png")
    static_red = cv2.imread("Data/Displays/box-red.png")

    blue = cv2.resize(blue, (1600, 900), interpolation=cv2.INTER_AREA)

    M = np.array([
        [0.968439, -0.00260381, -3.06587],
        [0.00439661, 0.902977, 13.6963]
    ])
    registered = cv2.warpAffine(blue, M, (1600, 900))

    cv2.imwrite('REGISTREDTEST.png', registered)

    cv2.imshow('registered', registered)
    cv2.waitKey()
    cv2.destroyAllWindows()

    #M = find_homography(moving=blue, static=static_blue)


    print(M)



    #cv2.imshow('red', static_red)
    #cv2.waitKey()



CHECKERBOARD = 1
BLEND = 2
def Composite(image1, image2, method=CHECKERBOARD, grid=(9, 16)):
    """
    compares two images
    :param image1:
    :param image2:
    :param method:
    :param grid: a 2-tuple, (rows, columns), that determines the shape of the checkerboard pattern.
    :return:

    NOTE: currently only checkerboard is implemented
    """
    ##########################################################################
    # Make sizes match
    ##########################################################################
    #transform the images to the same size. Resizes bigger to match smaller if different dimensions
    if (image1.shape[0] != image2.shape[0]) or (image1.shape[1] != image2.shape[1]): # if there is some difference in width or height

        if image1.shape[0] * image1.shape[1] > image2.shape[0] * image2.shape[1]: # if image1 is bigger
            image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]), interpolation=cv2.INTER_AREA)
        else: #image2 is bigger
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_AREA)

    ##########################################################################
    # Checkerboard
    ##########################################################################
    if method == CHECKERBOARD:
        #check for 1 of 3 cases, both color, both grayscale, or disparate.
        flag_gray = len(image1.shape) == 2 and len(image2.shape) == 2
        flag_color = len(image1.shape) == 3 and len(image2.shape) == 3
        flag_disparate = not (flag_gray or flag_color)

        #logical check that only 1 of the flags is active.
        assert sum([flag_gray, flag_color, flag_disparate]) == 1, "Error in logic for setting flags that check color mode of images being compared."

        #handle disparate. "bloom up" the grayscale one.
        if flag_disparate:
            if len(image1.shape) == 2: # image1 is grayscale
                image1 = np.stack((image1,)*3, axis=-1)
            else: # image2 is grayscale
                image2 = np.stack((image2,)*3, axis=-1)


        #Images should have same dimensions at this point
        assert image1.shape == image2.shape, "Error, images do not have the same dimensions."

        # Begin Tiling routine here.
        h, w = image1.shape[:2] # height and width of image
        rows, cols = grid  # dimensions of the tiling.

        # Calculate height and width of each tile
        tile_h = h // rows
        tile_w = w // cols

        # Initialize output image
        out = np.zeros_like(image1)

        #iterate over and copy each tile
        for r in range(rows):
            for c in range(cols):
                # Determine which image to use based on checkerboard pattern
                use_img1 = (r + c) % 2 == 0
                src = image1 if use_img1 else image2

                # Compute the slice for this tile
                y_start = r * tile_h
                y_end = (r + 1) * tile_h
                x_start = c * tile_w
                x_end = (c + 1) * tile_w

                # Copy tile to result
                out[y_start:y_end, x_start:x_end] = src[y_start:y_end, x_start:x_end]

        return out


def FindTransform(src_pts, dest_pts):
    """
    Finds the 3x3 transformation matrix
    :param src_pts:
    :param dest_pts:
    :return:
    """
    A = []
    for (x, y), (xp, yp) in zip(src_pts, dest_pts):
        A.append([x, y, 1, 0, 0, 0, -x * xp, -y * xp])
        A.append([0, 0, 0, x, y, 1, -x * yp, -y * yp])

    A = np.array(A)
    b = dest_pts.flatten()
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    # Append h33 = 1
    H = np.append(x, 1).reshape(3, 3)

    return H




def namehere(moving, static):
    """
    Does a least squares transformation.
    :param moving:
    :param static:
    :return:
    """
    #








def DrawPointsToVerify(image1, pts1, image2, pts2):
    """
    This script will
    :return:
    """
    #convert grayscale to color copying the intensity over each channel. x -> (x, x, x)
    if len(image1.shape) == 2:
        image1 = np.stack((image1,) * 3, axis=-1)

    if len(image2.shape) == 2:
        image2 = np.stack((image2,) * 3, axis=-1)

    pass



def DrawPoints(image, pts, size=10):
    #Make grayscale into bgr colo space if needed.
    if len(image.shape) == 2:
        image = np.stack((image,) * 3, axis=-1)

    #make copy of the image to draw on
    out = image.copy()

    colors = Color.GenerateColors(len(pts))
    for (y, x), color in zip(pts, colors):
        cv2.circle(out, (int(x), int(y)), radius=size, color=color, thickness=-1)  # Green dots

    return out


def rearranger():
    good = []


def test_composite():
    moving = cv2.imread("Data/Mar14Tests/0015.jpg")
    static = cv2.imread("Data/Displays/box-black.png")

    dimensions = (3, 5)    # (5, 8)
    composite = Composite(moving, static, grid=dimensions)

    cv2.imshow('', composite)
    cv2.waitKey()
    cv2.destroyAllWindows()



def test_find_homography():
    pass

def test():
    #test_gftt()
    #test_images()
    #test_sitk()
    test_composite()
    test_find_homography()


if __name__ == "__main__":
    test()
