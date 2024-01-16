"""
Verify.py is contains the routines for checking the accuracy (overall correctness) of some arbitrary technique for
attempting to find obstructed regions in the display.


To calculate the ground truth


The data structure that represents the obstructed regions of the display is a 2d matrix stored as a 2d list.
A True entry means the corresponding region is unobstructed, False means it is obstructed.
The data structure is accessed with m[row][col] where m[0][0] is the top-left region.
"""

import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk
from ConfusionMatrix import ConfusionMatrix



def create_gt_array(gt, n_rows, n_cols, threshold=0.95):
    """
    This function produces as 2d boolean matrix of dimensions n_rows x n_cols that represents the ground truth of
    test image. Each entry corresponds to the region of the image if it was partitioned into the same number of rows
    and columns. An entry is true if it is unobstructed; false if obstructed.
    The matrix is accessed like so: m[row][col].

    The parameter, gt, is an image mask representing the ground truth. Black, (0,0,0), is an unobstructed area of
    the display, otherwise it is an obstruction. Typically, obstructed regions will be White (255, 255, 255). The
    ground truth mask does not have to be a 2d image, it can be 3d too.

    n_rows/n_cols is the number of rows/columns for the ground truth to be partitioned.

    threshold is what percent of a region's pixels need to be visible for that region to be considered unobstructed.
    E.g. if threshold is set to 0.70 and 80% of a regions pixels are visible, then it will be considered
    unobstructed.
    """
    #ensure specificity makes sense and is valid
    assert threshold <= 1.0, "ERROR, invalid threshold. Should be 0.0 < threshold <= 1.0"
    assert threshold > 0.0, "ERROR, invalid threshold. Should be 0.0 < threshold <= 1.0"
    if threshold < 0.7:
        print(f"WARNING, your threshold, {threshold}, is set to a very low value. You risk having obstructed "
              f"regions be considered unobstructed.")
    assert len(gt.shape) == 2 or len(gt.shape) == 3, "Your image must have either 2 or 3 dimensions."


    #Partition the image with respect to the parameters
    if len(gt.shape) == 3:
        gt_part = partition_image(gt[:, :, 0], n_rows, n_cols)
    else:
        gt_part = partition_image(gt, n_rows, n_cols)

    out = []
    for r in range(n_rows):
        row = []
        for c in range(n_cols):
            region = gt_part[r][c]
            percent_unobstructed_pixels = 1 - (np.count_nonzero(region) / region.size)
            row.append(percent_unobstructed_pixels > threshold)
        out.append(row)

    return out





def partition_image(img, n_rows, n_cols):
    """
    Splits the image up into a rectilinear grid of sub-images.
    :param img: The image to be partitioned.
    :param n_rows: Number of rows.
    :param n_cols: Number of columns.
    :return: A 2-d matrix of dimensions n_rows x n_cols where each entry is an image corresponding to a region
    on the original image.
    """
    rows = np.array_split(img, n_rows, axis=0)
    image_grid = []
    for row in rows:
        image_grid.append(np.array_split(row, n_cols, axis=1))
    return image_grid






"""
The following routines are for visualizing a grid of images.
resize_image is a helper routine to simply ensure the images are the appropriate size on the canvas.
"""
def resize_image(img, max_size):
    """Resize the image using OpenCV and return a Tkinter-compatible image."""

    # Calculate the new size maintaining the aspect ratio
    height, width = img.shape[:2]
    ratio = min(max_size[0] / width, max_size[1] / height)
    new_size = int(width * ratio), int(height * ratio)

    # Resize the image using OpenCV
    resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    # Convert the OpenCV image to a PIL image
    if len(resized_img.shape) == 2:  # Grayscale image
        img = Image.fromarray(resized_img, mode='L')
    else:  # Color image
        img = Image.fromarray(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))

    return ImageTk.PhotoImage(img)



def visualize_image_grid(image_grid):
    window = tk.Tk()
    window.title("Image Grid")
    #window.configure(background='green')

    n_rows = len(image_grid)
    n_cols = len(image_grid[0])

    # set the max width to 80% of the screen width /n cols
    screen_width = window.winfo_screenwidth()
    max_image_size = (int((0.8 * screen_width) / n_cols), int((0.8 * screen_width) / n_cols))

    for row in range(n_rows):
        for col in range(n_cols):

            # Convert the NumPy array to a PIL Image
            img = image_grid[row][col]

            # Resize the image while maintaining aspect ratio
            #img = resize_image(img, max_image_size)
            #img_tk = ImageTk.PhotoImage(img)
            img_tk = resize_image(img, max_image_size)


            # Create a label and place it in the grid
            label = tk.Label(window, image=img_tk)
            label.configure(background='white')
            label.image = img_tk  # Keep a reference, prevent garbage-collection
            label.grid(row=row, column=col)

    window.mainloop()




def visualize_region_matrix(img, mask_arr):
    width = img.shape[1]
    height = img.shape[0]
    rows = len(mask_arr)
    cols = len(mask_arr[0])

    # create a copy of the image
    out_image = img[:]

    # grab width of the window
    window_width = width / cols
    window_height = height / rows

    # helper for drawing x
    def draw_x(local_origin_x, local_origin_y):
        # points are in reading order, 1=tl, 2=tr, 3=bl, 4=tr
        pt1 = (int(0.25 * window_width + local_origin_x), int(0.25 * window_height + local_origin_y))
        pt2 = (int(0.75 * window_width + local_origin_x), int(0.25 * window_height + local_origin_y))
        pt3 = (int(0.25 * window_width + local_origin_x), int(0.75 * window_height + local_origin_y))
        pt4 = (int(0.75 * window_width + local_origin_x), int(0.75 * window_height + local_origin_y))

        cv2.line(out_image, pt1, pt4, (0, 0, 255), 2)
        cv2.line(out_image, pt3, pt2, (0, 0, 255), 2)

    # iterate over and draw Xs on the obstructed places
    for r in range(rows):
        for c in range(cols):
            if not mask_arr[r][c]:
                draw_x(c * window_width, r * window_height)

    # return the resulting image
    return out_image






def verify(measured, expected, threshold):
    """
    This function is the key routine in this script. Determines how well the obstructed region was predicted by
    comparing it to the ground truth.
    :param measured: A 2d boolean matrix of where the obstructions are measured to be. To reiterate,
    measured[r][c] = True, means that the region at row r and column c is NOT obstructed.
    :param expected: An image of the mask for the ground truth of where the obstructions are located.
    A black pixel, 0, means that pixel is NOT obstructed. Anything other than 0 means it is obstructed. Typically, it
    will be white, 255, but all non-zero pixels will be considered obstructed.
    :param threshold: What percent of a region's pixels need to be visible for that region to be considered
    unobstructed. E.g. if threshold is set to 0.70 and 80% of a regions pixels are visible, then that region will be
    considered unobstructed. Similarly, a threshold of 0.99 with a region where 96% of pixels are visible will be
    considered obstructed
    :return: A ConfusionMatrix object from which accuracy and more information can be directly accessed.
    """

    #determine number of rows and columns being worked with
    n_rows = len(measured)
    n_cols = len(measured[0])

    #create the ground truth matrix
    gt_matrix = create_gt_array(expected, n_rows, n_cols, threshold=threshold)

    #Instantiate ConfusionMatrix object to store the results
    confusion_matrix = ConfusionMatrix()

    #iterate over the corresponding regions and tally the results
    for r in range(n_rows):
        for c in range(n_cols):
            confusion_matrix.register(measured[r][c], gt_matrix[r][c])

    #End routine, return results
    return confusion_matrix






############################################################################################################
# TESTS
############################################################################################################

def test_splitting_arrays():
    """
    This function just tests the numpy functionality for partitioning arrays
    """
    img = cv2.imread("/Users/aidanlear/PycharmProjects/DYSCO-2024/Data/TestSet1/test1/cam-low-exposure.png")
    #cv2.imshow("Original Image", img)

    n_rows = 5
    rows = np.array_split(img, n_rows, axis=1)
    print("number of rows partitioned", len(rows))
    for i in range(len(rows)):
        cv2.imshow(f"row {i+1}", rows[i])
    cv2.waitKey()


def test_gt_vis():
    """
    This tests the visualize_region_matrix function that simply visualizes what regions of an image are
    obstructed.
    """
    measured = cv2.imread("/Users/aidanlear/PycharmProjects/DYSCO-2024/Data/TestSet1/test1/cam-low-exposure.png")
    gt = cv2.imread("/Users/aidanlear/PycharmProjects/DYSCO-2024/Data/TestSet1/test1/obstructionmask.png")
    gt_arr = create_gt_array(gt, 4, 7)
    img = visualize_region_matrix(measured, gt_arr)
    cv2.imshow("testing", img)
    cv2.waitKey()


def test_visualizer():
    """
    Tests the visualize_image_grid routine that shows a partitioned image with Tkinter.
    """
    img = cv2.imread("/Users/aidanlear/PycharmProjects/DYSCO-2024/Data/TestSet1/test1/cam-low-exposure.png")
    n_rows = 2
    n_cols = 3
    image_grid = partition_image(img, n_rows, n_cols)
    visualize_image_grid(image_grid)



def test_primary_verify_routine():
    """
    Tests the Verify(...) routine.
    """
    #the threshold being used for these tests
    thresh = 0.95

    #How many rows and columns being worked with
    n_rows = 4
    n_cols = 7

    #read in the ground truth mask
    gt_mask = cv2.imread("/Users/aidanlear/PycharmProjects/DYSCO-2024/Data/TestSet1/test1/obstructionmask.png")

    #Test data being used
    perfect = [
        [True, True, True, False, False, False, False],
        [True, True, True, True, False, False, False],
        [True, True, True, True, True, False, False],
        [True, True, True, True, True, False, False],
    ]

    slight_mistake = [
        [True, True, True, False, False, False, False],
        [True, True, False, True, False, False, False],
        [True, True, True, True, True, False, False],
        [True, True, True, True, True, False, False],
    ]

    more_mistakes = [
        [True, True, True, False, True, False, False],
        [True, True, False, True, False, True, False],
        [True, False, True, True, True, True, True],
        [False, False, True, True, True, False, False],
    ]

    #Running the tests.
    #Note: The accuracy of t1 should be better than t2, and t2 should be better than t3
    t1 = verify(perfect, gt_mask, thresh)
    t2 = verify(slight_mistake, gt_mask, thresh)
    t3 = verify(more_mistakes, gt_mask, thresh)

    print("----TEST1----")
    print(t1)
    print("\n\n----TEST2----")
    print(t2)
    print("\n\n----TEST3----")
    print(t3)



if __name__ == "__main__":
    test_primary_verify_routine()
    test_visualizer()





