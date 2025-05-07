"""
Handles color specific operations




----Possible Color processing methods----
Do the color matching thing

"""


import numpy as np
import cv2
import Support
from sklearn.neighbors import NearestCentroid
from Support import ShowImage, ShowImages

def GenerateColors(n, format='bgr'):
    """
    returns a list of n rgb tuples, where each color is interpolated along, like, a rainbow.
    :param n: Number of colors to interpolate
    :return: A list of colors interpolated along hue values in the rgb color space
    """
    hues = [(360//n) * i for i in range(n)]
    output_colors = []
    for h in hues:
        x_1 = (h/60) % 2
        x_2 = 1 - abs(x_1 - 1)
        x = int(x_2 * 255)
        if h < 60:
            rgb_vector = (255, x, 0)
        elif h < 120:
            rgb_vector = (x, 255, 0)
        elif h < 180:
            rgb_vector = (0, 255, x)
        elif h < 240:
            rgb_vector = (0, x, 255)
        elif h < 300:
            rgb_vector = (x, 0, 255)
        else:
            rgb_vector = (255, 0, x)

        if format == 'bgr':
            rgb_vector = rgb_vector[::-1]

        output_colors.append(rgb_vector)

    return output_colors


def _nearest(a, X):
    """
    :param a: some vector if size w.
    :param X: An n x w matrix. Is a list of n vectors.
    :return: The index in X of the nearest neighbor to a.

    Currently implemented using the O(n) naive algorithm.
    """
    #check the input is of correct dimensions. length of a must be equal to the width of X.
    assert len(a) == X.shape[1], "Incorrect dimensions when finding nearest pixel color."
    best_distance = np.inf # set best distance to positive infinite
    best_index = -1 #current best index

    #iterate over the rows of candidates
    for i, x in zip(range(X.shape[0]), X): # iterate over
        distance = np.linalg.norm(a - x)
        if distance < best_distance:
            best_index = i
    return best_index


def ColorPaletteMatch(image: np.ndarray, template: np.ndarray, debug=False):
    """
    :param image: input image
    :param template: the image to be matched to.
    :return: A copy of image where the colors match the nearest in the template's color palette.
    """
    if debug: print("----Matching Color Palette----")
    #Possible candidate colors from the template
    palette = np.unique(Support.FlattenToVoxels(template), axis=0)

    #Flattened output image
    output = Support.FlattenToVoxels(image)

    if debug:
        print(f"Matching {len(output)} pixels to a color palette of {len(palette)} candidate pixels.")
        print("Setting up classifier...")


    #setup nearest centroid classifier
    classifier = NearestCentroid(metric="euclidean")
    classifier.fit(palette, np.array(list(range(1, palette.shape[0] + 1))))
    classifier.fit(palette, np.array(list(range(palette.shape[0]))))


    #apply classifier. Returns the labels
    if debug: print("Making prediction...")
    labels = classifier.predict(output)

    #transform based on the labels
    if debug: print("Applying labels to original image...")
    output = palette[labels]

    #reshape back to original
    output = output.reshape(image.shape)
    if debug: print("done.")
    return output









def test_color_match():
    #grab unique rows in multidimmensional array -> np.unique(a, axis=0)
    # background = cv2.imread("Data/Mar14Tests/0015.jpg")

    #The Shapes teapot images
    #moving = cv2.imread("Data/Mar14Tests/0049.jpg")
    #static = cv2.imread("Data/Displays/ShapesTeapot.png")

    #The black box image
    #moving = cv2.imread("Data/Mar14Tests/0015.jpg")
    moving = cv2.imread("TEMPORARY.png")
    static = cv2.imread("Data/Displays/box-black.png")

    #debug with a smaller

    #out = Support.FlattenToVoxels(moving)

    #apply transformation
    out = ColorPaletteMatch(moving, static)

    #Show results
    ShowImages((static, out), ('original', 'matched'))






def test():
    test_color_match()

if __name__ == "__main__":
    test()
