"""
The plan for this function is to have utility routines that are helpful throughout the code
"""
import cv2
import numpy as np





def row_selection(cols, image):
    """
    Automatically selects the appropriate number of rows for down-sampling given the number of columns and the
    image to be down-sampled.
    :param cols:
    :param image:
    :return:
    """
    h, w, d = image.shape
    rows = round(cols * (h / w))
    return rows



def aabb(image):
    """
    Removes extra transparent space around object within image to create a new image of the axially aligned
    bounding box
    :param image:
    :return:
    """
    if image.shape[2] < 4:
        raise ValueError("The image array must have an alpha channel.")

        # Find the alpha channel
    alpha_channel = image[:, :, 3]

    # Identify rows and columns that contain non-transparent pixels
    rows_with_content = np.any(alpha_channel != 0, axis=1)
    cols_with_content = np.any(alpha_channel != 0, axis=0)

    # Find the bounding box of the non-transparent areas
    y_min, y_max = np.where(rows_with_content)[0][[0, -1]]
    x_min, x_max = np.where(cols_with_content)[0][[0, -1]]

    # Crop the image using numpy slicing
    cropped_image_array = image[y_min:y_max + 1, x_min:x_max + 1, :]

    return cropped_image_array







def scale_image(image, scaling_factor, interpolation=None):
    """
    Apply the following transformation, where s is the scaling_factor:
        s, 0, 0
        0, s, 0
        0, 0, 1
    :param image:
    :param scaling_factor:
    :param interpolation:
    :return:
    """
    #grab dimensions of the image
    h, w = image.shape[0], image.shape[1]

    #automatically choose an interpolation method if one is not given
    if interpolation is None:
        if scaling_factor< 1.0:  # scaling down
            interpolation = cv2.INTER_AREA
        else:  # scaling up
            interpolation = cv2.INTER_CUBIC

    #resize/scale the image
    return cv2.resize(image, (w, h), interpolation=interpolation)


x = cv2.imread("Obstructions/line.png")
print(x.shape)





