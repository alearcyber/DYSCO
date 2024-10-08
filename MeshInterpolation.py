"""
Steps for determining the color of some candidate pixel within a convex polygon.
 - First, determine location of the pixel in barycentric coordinates using the shoelace formula.
 - Use the barycentric coordinates to find the original normalized coordinates within the original image, P_o.
 - Find the neighbors relative to the P_o. Neighbors are the pixel centers that make up the square that contains P_o.
    Use the j = P_jn+1/2 formula.
 - Fetch the colors of the neighbors.
 - Express P_o as barycentric coordinates of the neighbors.
 - Use the barycentric coordinates relative to neighbors to interpolate the color at P_o.
 - Set the candidate pixel to that color, Done.


Note: The polygon mesh is required to be in
"""
import cv2
import numpy as np
import scipy
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import math
from FastTriangleInterpolation import *


def barycentric_coordinates(a, b, c, p):
    def triangle_area(x1, y1, x2, y2, x3, y3):
        return abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

    area_abc = triangle_area(a[0], a[1], b[0], b[1], c[0], c[1])
    lambda_a = triangle_area(p[0], p[1], b[0], b[1], c[0], c[1]) / area_abc
    lambda_b = triangle_area(a[0], a[1], p[0], p[1], c[0], c[1]) / area_abc
    lambda_c = triangle_area(a[0], a[1], b[0], b[1], p[0], p[1]) / area_abc

    return lambda_a, lambda_b, lambda_c







def MeshInterpolate(image, in_mesh, out_mesh, mesh2):
    """
    Primary Function,
    Interpolates...
    """








def InterpolateTriangleTest():
    """
    test the core of the method.
    Works very well.
    """
    image = cv2.imread("/Users/aidan/Desktop/penguin.png")
    out = image.copy()


    tri1 = [(300, 600), (500, 1300), (900, 500)]
    tri2 = [(300, 630), (500, 1650), (900, 500)]

    tri1 = np.array(tri1)
    tri2 = np.array(tri2)


    for l1 in np.arange(0, 1, 0.0005):
        for l2 in np.arange(0, 1 - l1, 0.0005):
            l3 = 1.0 - l1 - l2

            # barycentric coordinates, already found! boom
            # l1, l2, and l3


            # find coordinates within the original image
            original_coordinates = (l1 * tri1[0]) + (l2 * tri1[1]) + (l3 * tri1[2])



            #grab neighbors
            tl = int(original_coordinates[0]), int(original_coordinates[1])
            tr = tl[0] + 1, tl[1]
            bl = tl[0], tl[1] + 1
            br = tl[0] + 1, tl[1] + 1



            #fetch color located at each neighbor
            tl_color = image[tl[0], tl[1]]
            tr_color = image[tr[0], tr[1]]
            bl_color = image[bl[0], bl[1]]
            br_color = image[br[0], br[1]]

            #grab barycentric of point relative to neighbors
            lx = original_coordinates[0] - tl[0]
            ly = original_coordinates[1] - tl[1]

            #bilinear interpolation
            top_color = tl_color*(1-lx) + tr_color*lx
            bottom_color = bl_color*(1-lx) + br_color*lx
            candidate_color = top_color*(1-ly) + bottom_color*ly

            #applycolor
            color = np.array(candidate_color).astype(np.uint8)
            candidate_coordinates = (l1 * tri2[0]) + (l2 * tri2[1]) + (l3 * tri2[2])

            x, y = int(candidate_coordinates[0] + 0.5), int(candidate_coordinates[1] + 0.5)
            try:
                out[x, y] = color
            except IndexError:
                pass

    #draw points to see change
    for i in [0,1,2]:
        cv2.circle(image, (tri1[i][1],tri1[i][0]), 5, (200, 0, 0), -1)

    for i in [0,1,2]:
        cv2.circle(out, (tri2[i][1],tri2[i][0]), 5, (200, 0, 0), -1)



    cv2.imshow("original", image)
    cv2.imshow('result', out)
    cv2.waitKey()




def InterpolateQuad():
    """works, but slow"""
    image = cv2.imread("/Users/aidan/Desktop/penguin.png")
    out = image.copy()

    quad1 = np.array([(500, 700), (900, 700), (900, 1300), (500, 1300)])
    quad2 = np.array([(500, 700), (900, 700), (900 + 200, 1300 + 200), (500 - 200, 1300 + 200)])

    for l1 in np.arange(0, 1, 0.005):
        for l2 in np.arange(0, 1 - l1, 0.005):
            for l3 in np.arange(0, 1.0 - l1 - l2, 0.005):
                l4 = 1.0 - l1 - l2 - l3

                # barycentric coordinates, already found! boom
                # l1, l2, l3, and l4

                # find coordinates within the original image
                original_coordinates = (l1 * quad1[0]) + (l2 * quad1[1]) + (l3 * quad1[2]) + (l4 * quad1[3])

                # grab neighbors
                tl = int(original_coordinates[0]), int(original_coordinates[1])
                tr = tl[0] + 1, tl[1]
                bl = tl[0], tl[1] + 1
                br = tl[0] + 1, tl[1] + 1

                # fetch color located at each neighbor
                tl_color = image[tl[0], tl[1]]
                tr_color = image[tr[0], tr[1]]
                bl_color = image[bl[0], bl[1]]
                br_color = image[br[0], br[1]]

                # grab barycentric of point relative to neighbors
                lx = original_coordinates[0] - tl[0]
                ly = original_coordinates[1] - tl[1]

                # bilinear interpolation
                top_color = tl_color * (1 - lx) + tr_color * lx
                bottom_color = bl_color * (1 - lx) + br_color * lx
                candidate_color = top_color * (1 - ly) + bottom_color * ly

                # applycolor
                color = np.array(candidate_color).astype(np.uint8)
                candidate_coordinates = (l1 * quad2[0]) + (l2 * quad2[1]) + (l3 * quad2[2]) + (l4 * quad2[3])

                x, y = int(candidate_coordinates[0] + 0.5), int(candidate_coordinates[1] + 0.5)
                try:
                    out[x, y] = color
                except IndexError:
                    pass

    # draw points to see change
    for i in [0, 1, 2, 3]:
        cv2.circle(image, (quad1[i][1], quad1[i][0]), 5, (200, 0, 0), -1)

    for i in [0, 1, 2, 3]:
        cv2.circle(out, (quad2[i][1], quad2[i][0]), 5, (200, 0, 0), -1)

    cv2.imshow("original", image)
    cv2.imshow('result', out)
    cv2.waitKey()




def InterpolateTriangle(source, out, in_vertices, out_vertices):
    """
    Make it work with arguments
    """
    tri1 = in_vertices
    tri2 = out_vertices


    for l1 in np.arange(0, 1, 0.005):
        for l2 in np.arange(0, 1 - l1, 0.005):
            l3 = 1.0 - l1 - l2

            # barycentric coordinates, already found! boom
            # l1, l2, and l3


            # find coordinates within the original image
            original_coordinates = (l1 * tri1[0]) + (l2 * tri1[1]) + (l3 * tri1[2])



            #grab neighbors
            tl = int(original_coordinates[0]), int(original_coordinates[1])
            tr = tl[0] + 1, tl[1]
            bl = tl[0], tl[1] + 1
            br = tl[0] + 1, tl[1] + 1

            #TODO - fix edge case here
            try:
                #fetch color located at each neighbor
                tl_color = source[tl[0], tl[1]]
                tr_color = source[tr[0], tr[1]]
                bl_color = source[bl[0], bl[1]]
                br_color = source[br[0], br[1]]
            except IndexError:
                continue

            #grab barycentric of point relative to neighbors
            lx = original_coordinates[0] - tl[0]
            ly = original_coordinates[1] - tl[1]

            #bilinear interpolation
            top_color = tl_color*(1-lx) + tr_color*lx
            bottom_color = bl_color*(1-lx) + br_color*lx
            candidate_color = top_color*(1-ly) + bottom_color*ly

            #applycolor
            color = np.array(candidate_color).astype(np.uint8)
            candidate_coordinates = (l1 * tri2[0]) + (l2 * tri2[1]) + (l3 * tri2[2])

            x, y = int(candidate_coordinates[0] + 0.5), int(candidate_coordinates[1] + 0.5)
            try:
                out[x, y] = color
            except IndexError:
                pass







def draw_dots():
    image = cv2.imread("/Users/aidan/Desktop/penguin.png")
    blue, red, green = (200, 0, 0), (0,0,200), (0,200,0)

    #triangle 1
    tri1 = [(600, 300),
              (1300, 500),
              (500, 900)]
    for point in tri1:
        cv2.circle(image, point, 5, blue, -1)


    #triangle 2
    tri2 = [(630, 300),
            (1330, 530),
            (500, 900)]

    for point in tri2:
        cv2.circle(image, point, 5, red, -1)




    cv2.imshow("", image)
    cv2.waitKey()



def draw_mesh(image, mesh, triangulation):
    #image dimensions
    n_rows = image.shape[0]

    #build vertex list to draw
    vertices = []
    for triangle in triangulation.simplices:
        # parse out the edges
        edges = [(triangle[0], triangle[1]), (triangle[0], triangle[2]), (triangle[1], triangle[2])]
        edges_reversed = [(triangle[1], triangle[0]), (triangle[2], triangle[0]), (triangle[2], triangle[1])]

        #iterate over edges
        for i in range(3):
            if (edges[i] not in vertices) and (edges_reversed not in vertices):
                vertices.append(edges[i])

    #draw edges
    for vertex in vertices:
        y1, x1 = mesh[vertex[0]]
        y2, x2 = mesh[vertex[1]]
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 0), 2)


    #draw vertices
    for y, x in mesh:
        cv2.circle(image, (x, y), 5, (200, 0, 0), -1)

    #done

def test_drawing_mesh():
    # create triangulation
    mesh = np.array([(300, 600), (500, 1300), (900, 500), (0, 0), (1467, 0), (0, 2200), (1467, 2200)])
    triangulation = Delaunay(mesh)

    #read in image
    image = cv2.imread("/Users/aidan/Desktop/penguin.png")
    original = image.copy()


    #draw mesh
    draw_mesh(image, mesh, triangulation)



    #show resulting images
    cv2.imshow('original', original)
    cv2.imshow('with mesh', image)
    cv2.waitKey()


def rasterize(v1, v2, v3, i1, i2, i3):
    """
    t is for triangle. It is a list of verteces

    Assumes vertices are in order of peak, left, right
    """

    ###################################################
    # determine scanline positions
    ###################################################

    # first find highest and lowest y-values
    #maximum = max([v1[1], v2[1], v3[1]])
    #minimum = min([v1[1], v2[1], v3[1]])
    #maximum = int(maximum)
    #minimum = math.ceil(minimum)
    if v1[1] > v2[1]:
        maximum, minimum = int(v1[1]), math.ceil(v2[1])
    else:
        maximum, minimum = int(v2[1]), math.ceil(v1[1])


    # now I can use those values to get the y values I need to scan over
    raster_positions = [i for i in range(minimum, maximum + 1)]



    ###################################################
    # Interpolate along left and right edges
    ###################################################
    #call helper function to interpolate the X coordinates
    X_left, I_left = _interpolate_edge(raster_positions, v1, v2, i1, i2)
    X_right, I_right = _interpolate_edge(raster_positions, v1, v3, i1, i3)



    ###################################################
    # Interpolate along left and right edges
    ###################################################
    #iterate over the scan lines
    for y, x_left, i_left, x_right, i_right in zip(raster_positions, X_left, I_left, X_right, I_right):

        #iterate across each scan lines
        for x in range(int(x_left + 1.0), int(x_right + 1.0)):
            #find x in parametric coordinates
            l = (x - x_left) / (x_right - x_left)

            #interpolate intensity
            intensity = (i_left*(1-l)) + (i_right*l)

            #do what is needed with the intensity here.


    print(raster_positions)




def _determine_y_values(p1, p2, p3, i1, i2):
    """
    Determines the scan line heights for a given triangle. Returns a list of the scan lines.
    """
    #first find highest and lowest y-values
    maximum = max([p1[1], p2[1], p3[1]])
    minimum = min([p1[1], p2[1], p3[1]])

    #biggest round down, smallest round up
    maximum = int(maximum)
    minimum = math.ceil(minimum)

    #find first x values

    #now I can use those values to get the y values I need to scan over
    raster_positions = [i for i in range(minimum, maximum+1)]

    #return raster positions
    return raster_positions


def _interpolate_edge(scan_lines, v1, v2, i1, i2):
    #determine lambda at first scan line
    y = scan_lines[0]
    l = (y - v1[1]) / (v2[1] - v1[1])
    I = [(i1 * (1 - l)) + (i2 * l)]

    #use lambda to determine x-value of first line to scan
    x = v1[0]*(1-l) + v2[0]*l
    X = [x]  #place to store output values

    #calculate change to x for every subsequent scan line, inverse slope
    dx = (v2[0] - v1[0])/(v2[1] - v1[1])


    #iterate over and calculate other scan lines
    i = 1
    while i < len(scan_lines):
        #calculate x
        x = x + dx
        X.append(x)

        #calculate lambda and interpolate intensity
        l += 1 / (v2[1] - v1[1])
        out_intensity = (i1 * (1 - l)) + (i2 * l)
        I.append(out_intensity)

        i += 1

    #return list of X values and interpolated intensities along the edge
    return X, I




def scanline_tri_inter(v1, v2, v3):
    pass









def test_scipy_inter():

    """
    THIS ONE WORKS.
    Need to make this more robust, like, need to make it a bit faster
    """
    image = cv2.imread("/Users/aidan/Desktop/penguin.png")
    cv2.imshow('original', image)
    out = np.copy(image)

    mesh1 = np.array([(300, 600), (500, 1300), (900, 500), (0, 0), (1467, 0), (0, 2200), (1467, 2200)])
    mesh2 = np.array([(300, 630), (500, 1650), (900, 500), (0, 0), (1467, 0), (0, 2200), (1467, 2200)])

    #create mesh
    tri = Delaunay(mesh1)


    #visualize original with mesh overlapped
    original_with_mesh = image.copy()
    draw_mesh(original_with_mesh, mesh1, tri)
    cv2.imshow("original mesh", original_with_mesh)
    cv2.imwrite('ORIGINAL_MESH.png', original_with_mesh)



    for vertex_indices in tri.simplices:
        #grab triangles
        v1, v2, v3 = vertex_indices[0], vertex_indices[1], vertex_indices[2]
        in_triangle = np.array([mesh1[v1], mesh1[v2], mesh1[v3]])
        out_triangle = np.array([mesh2[v1], mesh2[v2], mesh2[v3]])

        #interpolate the triangle
        InterpolateTriangle(image, out, in_triangle, out_triangle)

    #draw mesh on output copy
    out_with_mesh = out.copy()
    draw_mesh(out_with_mesh, mesh2, tri)
    cv2.imshow('result with mesh', out_with_mesh)
    cv2.imwrite('RESULT_WITH_MESH.png', out_with_mesh)

    #show result
    cv2.imshow('result', out)
    cv2.imwrite('RESULT.png', out)
    cv2.waitKey()


"""
img1 = cv2.imread("/Users/aidan/Desktop/penguin.png")
imgcopy = img1.copy()
InterpolateTriangle(img1, imgcopy, np.array([(300, 600), (500, 1300), (900, 500)]), np.array([(300, 630), (500, 1650), (900, 500)]))
cv2.imshow('result', imgcopy)
cv2.waitKey()
"""







#InterpolateTriangleTest()
#InterpolateQuad()
#test_scipy_inter()
#draw_mesh()
#test_drawing_mesh()
#triangle = ((1.2,1.2),(3.1, 7.6),(7.9, 1.2))
#triangle = ((3.1, 7.6), (1.2, 1.2), (7.9, 1.2))
#intensities = (0.0, 100.0, 50.0)
#rasterize(*triangle, *intensities)







"""
Forbidden Fruit
"""
def ff():
    import matplotlib.pyplot as plt

    #(3.1, 7.6), (1.2, 1.2), (7.9, 1.2)
    v0 = np.array([1.2, 1.2])
    v1 = np.array([3.8, 7.6])
    v2 = np.array([10.9, 7.6])
    v3 = np.array([14.6, 1.2])

    plt.plot(*v0, *v1, 'bo')  # plot x and y using blue circle markers
    plt.plot(*v1, *v2, 'bo')
    plt.plot(*v2, *v3, 'bo')
    plt.plot(*v3, *v0, 'bo')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)




    # Function to compute bilinear interpolation given barycentric coordinates and vertex values
    def bilinear_interpolation(bc, values):
        print(bc.T, values)
        return bc.T @ values


    # Function to compute barycentric coordinates of a point (px, py) in a quadrilateral
    def compute_barycentric(px, py, v0, v1, v2, v3):
        # Setup Ax=b
        A = np.array([
            [v0[0], v1[0], v2[0], v3[0]],
            [v0[1], v1[1], v2[1], v3[1]],
            [1, 1, 1, 1]
        ])
        b = np.array([[px, py, 1]]).T


        # Solve linear system.
        # lstsq returns a 4-tuple of (x, residuals, rank, svd_of_A). I only need x, labeled l for lambda in this case.
        l, _, _, _ = np.linalg.lstsq(A, b, rcond=-1)

        return l


    # Values at vertices (for example purposes, use values of your choice)
    values = np.array([v0, v1, v2, v3]).T


    # Get the bounding box of the quadrilateral
    min_x = int(min(v0[0], v1[0], v2[0], v3[0]))
    max_x = int(max(v0[0], v1[0], v2[0], v3[0]))
    min_y = int(min(v0[1], v1[1], v2[1], v3[1]))
    max_y = int(max(v0[1], v1[1], v2[1], v3[1]))

    # Iterate over each pixel in the bounding box
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            # Compute barycentric coordinates
            #bc = compute_barycentric(x + 0.5, y + 0.5, v0, v1, v2, v3)  # Using pixel center
            bc = compute_barycentric(x, y, v0, v1, v2, v3)  # Using pixel center


            print("BC SHAPE:", bc.shape)
            # Check if the pixel is inside the quadrilateral
            if all(0 <= bc) and all(bc <= 1):
                # Perform bilinear interpolation
                interpolated_value = values @ bc
                print("TPOSE:", interpolated_value.T)
                x, y = interpolated_value.T[0]
                plt.plot(x, y, 'k.')
                #plt.plot(*v0, *v1, 'bo')
                # Do something with the interpolated value (e.g., store or process it)
                print(f"Pixel ({x}, {y}): Interpolated Value = {interpolated_value}")
    plt.show()

#ff()



"""
Bringing it all together for a single triangle

So I have the triangle, barycentric coordinates, and the location of a pixel I want to light.
1) Grab triangle coordinates in the same triangle in the original image
2) Interpolate the barycentric coordinates relative to the original triangle coordinates to find the original position in sub-pixel accuracy.
3) calculate the position of the neighbors.
4) retrieve intensity of each neighbor.
5) Find candidate color with Bilinear interpolation of original triangle coordinates and intensity of each neighbor.
6) Light candidate pixel.
"""

def interpolate_triangle(image, source, triangle, source_triangle):
    """
    :param image: image to be modified
    :param source: source image.
    :param triangle: triangle within the image being modified
    :param source_triangle: triangle within the source image
    """
    #grab axially-aligned bounding box of the triangle
    v0, v1, v2 = triangle
    min_x = int(min(v0[0], v1[0], v2[0]))
    max_x = int(max(v0[0] + 1, v1[0] + 1, v2[0] + 1))
    min_y = int(min(v0[1], v1[1], v2[1]))
    max_y = int(max(v0[1] + 1, v1[1] + 1, v2[1] + 1))

    #construct matrix of source triangle
    source_triangle = np.array(source_triangle).T

    rows, cols = source.shape[0], source.shape[1]

    # Iterate over each pixel in the bounding box
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):

            # Compute barycentric coordinates
            bc = compute_barycentric(x, y, v0, v1, v2)

            # only compute if point is within the triangle
            if all(0 <= bc) and all(bc <= 1):
                #2) interpolate the location within the original triangle using the barycentric coordinates
                original_location = source_triangle @ bc
                x_o, y_o = original_location[0, 0], original_location[1, 0] # un-pack from numpy array

                #3) Calculate position of the neighbors
                x1, x2, y1, y2 = int(x_o), int(x_o + 1), int(y_o), int(y_o + 1)

                #4) Retrieve intensities of each neighbors, handle neighbor being out of bounds with mirror
                """
                if y2 >= cols:
                    y2 = cols - y2 - 1
                if x2 >= rows:
                    x2 = rows - x2 - 1
                if y1 < 0:
                    y1 = abs(y1) - 1
                if x1 < 0:
                    x1 = abs(x1) - 1
                if x1 == 2200:
                    print("HERE:", x1)
                if y1 == 2200:
                    print("rows:", rows)
                    print("cols:", cols)
                    print("HERE2:", y1)
                """
                try:

                    #4
                    q11, q12, q21, q22 = source[x1, y1], source[x1, y2], source[x2, y1], source[x2, y2]

                    # 5) Bilinear interpolate to find intensity
                    candidate_intensity = unitcube_bilinear_interpolation((x, y), q11, q12, q21, q22)

                    # 6) Light candidate pixel.
                    image[x, y] = candidate_intensity
                except IndexError:
                    pass





                #print("TPOSE:", interpolated_value.T)
                #x, y = interpolated_value.T[0]
                #plt.plot(x, y, 'k.')
                # plt.plot(*v0, *v1, 'bo')
                # Do something with the interpolated value (e.g., store or process it)
                #print(f"Pixel ({x}, {y}): Interpolated Value = {interpolated_value}")




def test_interpolate_triangle():
    #setup input
    tri1 = [(300, 600), (500, 1300), (900, 500)]
    tri2 = [(300, 630), (500, 1650), (900, 500)]
    image = cv2.imread("/Users/aidan/Desktop/penguin.png")
    out = image.copy()

    #call routine
    interpolate_triangle(out, source=image, triangle=tri2, source_triangle=tri1)

    #show results
    cv2.imshow("", out)
    cv2.waitKey()


def with_more():
    mesh1 = np.array([(300, 600), (500, 1300), (900, 500), (0, 0), (1467, 0), (0, 2200), (1467, 2200)])
    mesh2 = np.array([(300, 630), (500, 1650), (900, 500), (0, 0), (1467, 0), (0, 2200), (1467, 2200)])
    image = cv2.imread("/Users/aidan/Desktop/penguin.png")
    out = image.copy()

    # create mesh
    tri = Delaunay(mesh1)

    #iterate over each triangle and interpolate
    i = 0
    for vertex_indices in tri.simplices:
        #grab triangles
        v1, v2, v3 = vertex_indices[0], vertex_indices[1], vertex_indices[2]
        in_triangle = np.array([mesh1[v1], mesh1[v2], mesh1[v3]])
        out_triangle = np.array([mesh2[v1], mesh2[v2], mesh2[v3]])

        #interpolate the triangle
        #InterpolateTriangle(image, out, in_triangle, out_triangle)
        interpolate_triangle(out, source=image, triangle=out_triangle, source_triangle=in_triangle)

        print(i := i+1)


    # show results
    cv2.imshow("", out)
    cv2.waitKey()




def mesh_with_keypoints():

    import SelectingKeypoints

    test_number = 1
    IMAGE = cv2.imread(f"Data/TestingDiffDiff/test{test_number}/unobstructed-aligned.png")
    EXPECTED = cv2.imread(f"Data/TestingDiffDiff/test{test_number}/expected.png")
    out = IMAGE.copy() #where output image will be drawn

    # Find matching keypoints in each region
    kps, desc, kps_other, desc_other = SelectingKeypoints.strongest_match_each_region(IMAGE, EXPECTED, row_stride_count=10, col_stride_count=15, lowe_ratio=0.75)

    #Add corners of image to each.
    corners = [(0,0), (IMAGE.shape[0], 0), (0, IMAGE.shape[1]), (IMAGE.shape[0], IMAGE.shape[1])]
    kps += corners
    kps_other += corners

    #create mesh from keypoints
    triangulation = Delaunay(np.array(kps_other))

    # iterate over each triangle and interpolate
    i = 1
    for vertex_indices in triangulation.simplices:
        # grab triangles
        v1, v2, v3 = vertex_indices[0], vertex_indices[1], vertex_indices[2]
        in_triangle = np.array([kps_other[v1], kps_other[v2], kps_other[v3]])
        out_triangle = np.array([kps[v1], kps[v2], kps[v3]])

        # interpolate the triangle
        # InterpolateTriangle(image, out, in_triangle, out_triangle)
        interpolate_triangle(out, source=IMAGE, triangle=in_triangle, source_triangle=out_triangle)

        print(i)
        i += 1


    cv2.imwrite("THISISIT3.png", out)
    print("DONE!")


def now_see_the_results():
    expected = cv2.imread("/Users/aidan/PycharmProjects/DYSCO/Data/TestingDiffDiff/test1/expected.png")
    observed1 = cv2.imread("/Users/aidan/PycharmProjects/DYSCO/Data/TestingDiffDiff/test1/unobstructed-aligned.png")
    observed2 = cv2.imread("THISISIT.png")
    observed3 = cv2.imread("THISISIT2.png")

    def diff(img1, img2):
        result = np.linalg.norm(img1.astype(np.float64) - img2.astype(np.float64), axis=2)
        result = result * (255.0 / 441.673)
        return result.astype(np.uint8)

    original_diff = diff(expected, observed1)
    diff2 = diff(expected, observed2)
    diff3 = diff(expected, observed3)


    cv2.imshow("og", original_diff)
    cv2.imshow("2", diff2)
    cv2.imshow("3", diff3)
    cv2.waitKey()

mesh_with_keypoints()
#now_see_the_results()



