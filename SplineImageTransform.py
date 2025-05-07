import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline
import cv2

#image = cv2.imread("/Users/aidanlear/PycharmProjects/DYSCO-2024/GeneratedSet/dash6-obstruction0.png")


def draw_pointcloud():
    #control points
    points = np.array([
        [[1, 10], [5, 8], [7, 9], [12, 7]],
        [[1, 7], [4.5, 5], [7, 6], [11.25, 4]],
        [[1, 4], [4, 2], [7.5, 3], [10.5, 1]],
        [[1, 1], [3.5, -0.5], [8, 0], [9.5, -1]]
    ])
    degree = 3 # cubic
    t = np.array([0, 0, 0, 0, 1, 1, 1, 1]) #knots
    for r in range(points.shape[0]):
        for c in range(points.shape[1]):
            p = points[r, c]
            plt.scatter(p[0], p[1], c="red", s=14)



    #construct sparse point cloud from spline
    h_splines = [BSpline(t, row, degree) for row in points]
    for x in np.linspace(0, 1, 20):
        v_points = np.array([spline(x) for spline in h_splines])
        v_spline = BSpline(t, v_points, degree)
        for y in np.linspace(0, 1, 20):
            p = v_spline(y)
            plt.scatter(p[0], p[1], c="black", s=4)

    plt.show()



def interpolate_image():
    #image
    image = cv2.imread("/Users/aidanlear/PycharmProjects/DYSCO-2024/GeneratedSet/dash6-obstruction0.png")

    #spline params
    dx = 90
    dy = 110
    """
    points = np.array([  # control points
        [[1*dx, 10], [5, 8], [7, 9], [12, 7]],
        [[1, 7], [4.5, 5], [7, 6], [11.25, 4]],
        [[1, 4], [4, 2], [7.5, 3], [10.5, 1]],
        [[1, 1], [3.5, -0.5], [8, 0], [9.5, -1]]
    ])
    """
    points = np.array([  # control points
        [[1*dx, 10*dy], [5*dx, 8*dy], [7*dx, 9*dy], [12*dx, 7*dy]],
        [[1*dx, 7*dy], [4.5*dx, 5*dy], [7*dx, 6*dy], [11.25*dx, 4*dy]],
        [[1*dx, 4*dy], [4*dx, 2*dy], [7.5*dx, 3*dy], [10.5*dx, 1*dy]],
        [[1*dx, 1*dy], [3.5*dx, -0.5*dy], [8*dx, 0*dy], [9.5*dx, -1*dy]]
    ])
    degree = 3  # cubic
    t = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # knots

    # hold horizontal splines
    h_splines = [BSpline(t, row, degree) for row in points]

    #subroutine to interpolate
    def _interpolate(lambda_row, lambda_column):
        v_points = np.array([spline(lambda_column) for spline in h_splines])
        v_spline = BSpline(t, v_points, degree)
        return v_spline(lambda_row)


    #create a transformation map.
    #source pixel at (row, col) is transformed to transformation_map[row, col]
    #so if map[16, 20] = is (50.1, 67.2), that means the source pixel at (16,20) is transformed to (50.1, 67.2)
    transformation_map = np.empty((image.shape[0], image.shape[1], 2), dtype=np.float64)


    #iterate over every pixel in source to construct transformation map
    height = image.shape[0]
    width = image.shape[1]
    for row in range(height):
        for col in range(width):
            #pixel = image[row, col]
            l1, l2 = row /(height-1), col/(width-1) #barycentric coordinates
            transformation_map[row, col] = _interpolate(l1, l2)

    print(transformation_map.shape)


    #find the min and max resulting coordinates to allocate appropriate space for output image
    x_min = np.amin(transformation_map[:, :, 0])
    x_max = np.amax(transformation_map[:, :, 0])
    y_min = np.amin(transformation_map[:, :, 1])
    y_max = np.amax(transformation_map[:, :, 1])
    print("minimum:", (x_min, y_min))
    print("maximum:", (x_max, y_max))
    print("done")


    #apply transform to each pixel
    vertical_offset = 120
    transformed_image = np.empty((1100, 1300, 3), dtype=np.uint8)
    for r in range(transformation_map.shape[0]):
        for c in range(transformation_map.shape[1]):
            dest = transformation_map[r, c]
            pixel = image[r, c]
            dest_r = dest[0] + vertical_offset
            dest_c = dest[1]
            transformed_image[dest_r, dest_c] = pixel


    cv2.imshow('', transformed_image)
    cv2.waitKey(0)




#image = cv2.imread("/Users/aidanlear/PycharmProjects/DYSCO-2024/GeneratedSet/dash6-obstruction0.png")
#print(image.shape)

interpolate_image()





