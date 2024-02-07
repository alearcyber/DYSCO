import random
import cv2
import numpy as np
import Dysco


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





def overlay_image(base_image, overlay_image, position):
    """
    Overlay an image on top of another at a specified position, allowing the overlay image
    to be partially out of bounds.

    Parameters:
    - base_image: The larger image as a numpy array on which to overlay the smaller image.
    - overlay_image: The smaller image as a numpy array to be placed on top of the base image.
    - position: A tuple (x, y) specifying the top-left corner where the overlay image will be placed on the base image.

    Returns:
    - A numpy array representing the base image with the overlay image placed on top, with out-of-bounds parts omitted.
    """
    x, y = position  # Top-left position for overlay

    # Calculate the overlay's bounding box within the base image
    y1, x1 = max(y, 0), max(x, 0)  # Start coords (clamped to base image bounds)
    y2 = min(y + overlay_image.shape[0], base_image.shape[0])  # End coord Y
    x2 = min(x + overlay_image.shape[1], base_image.shape[1])  # End coord X

    # Calculate the region of the overlay to use
    overlay_y1 = max(-y, 0)
    overlay_x1 = max(-x, 0)
    overlay_y2 = overlay_y1 + (y2 - y1)
    overlay_x2 = overlay_x1 + (x2 - x1)

    # If there's nothing to overlay
    if y2 <= y1 or x2 <= x1:
        return base_image  # No change if overlay is completely out of bounds

    # Overlay
    if overlay_image.shape[2] == 4:  # With alpha channel
        alpha = overlay_image[overlay_y1:overlay_y2, overlay_x1:overlay_x2, 3] / 255.0
        for c in range(3):  # For each color channel
            base_image[y1:y2, x1:x2, c] = (alpha * overlay_image[overlay_y1:overlay_y2, overlay_x1:overlay_x2, c] +
                                           (1 - alpha) * base_image[y1:y2, x1:x2, c])
    else:  # No alpha channel
        base_image[y1:y2, x1:x2] = overlay_image[overlay_y1:overlay_y2, overlay_x1:overlay_x2]

    return base_image



#parameters to consider for the obstruction
# horizontal flipping
# vertical flipping
# scaling by some factor. 0.75 to 1.25, plus or minus 25% in size
# location:



def randomly_place_obstruction(base, overlay):
    """
    bounded such that a maximum of half the width or height of the object could be cut off, like the outer bounds.
    """
    #calculate outer boundary to place the overlay
    H, W, _ = base.shape
    h, w, _ = overlay.shape
    x_max = int(W - (0.5 * w))
    x_min = int(-0.5*w)
    y_max = int(H - (0.5 * h))
    y_min = int(-0.5*h)

    #generate location within outer boundary
    p = (random.randint(x_min, x_max), random.randint(y_min, y_max))

    #place the obstruction at that location using overlay_image routine
    return overlay_image(base, overlay, p)


def create_modification_mask(original_image, modified_image):
    """
    Create a mask indicating where pixels have been modified between two images.

    Parameters:
    - original_image: A numpy array representing the original image.
    - modified_image: A numpy array representing the modified version of the original image.

    Returns:
    - A numpy array representing the mask, where modified pixels are white (255, 255, 255)
      and unchanged pixels are black (0, 0, 0).
    """
    if original_image.shape != modified_image.shape:
        raise ValueError("Original and modified images must have the same dimensions and color depth.")

    # Compare the two images to find where pixels have changed
    difference = np.not_equal(original_image, modified_image)

    # Since `difference` is a boolean array, we need to convert it to uint8 and multiply by 255 to get the mask
    # We use any along the last axis (color channels) to check if any of the channels are different
    mask = np.any(difference, axis=-1).astype(np.uint8) * 255

    # Expand the mask to 3 channels to make it RGB (black & white)
    mask_rgb = np.stack([mask, mask, mask], axis=-1)

    return mask_rgb







def randomly_transform_obstruction(image):
    """
    applies random transformation to the image.

    The transformations are limited to combinations of the following:
        - flip vertical (or not)
        - flip horizontal (or not)
        - Scale size by a factor of s where s is [0.75,1.25]
        - rotation by 0, 90, 180, 270 or degrees

    """
    #Make a copy of the original image to accumulate output
    out = np.copy(image)


    #Randomly generate transformation parameters
    flip_vertical = random.choice([True, False])
    flip_horizontal = random.choice([True, False])
    scale = random.randint(75, 125) / 100
    rotation = random.choice([None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE])

    #first, apply flipping
    if flip_vertical:
        out = np.fliplr(out)
    if flip_horizontal:
        out = np.flipud(out)


    #scale the image
    out = Dysco.scale_image(out, scale)

    #rotate the image
    if rotation is not None:
        out = cv2.rotate(image, rotation)

    #return the resulting aggregate transformed image.
    return out












def test_randomly_placing_obstructions(iters):
    """
    Just a test that randomly places some obstructions on an image a bunch just
    to see what the behavior is, and to demonstrate the routine itself.
    :param iters:
    """
    #list of obstructed overlays
    obstructions = [
        cv2.imread("Obstructions/corner.png", cv2.IMREAD_UNCHANGED),
        cv2.imread("Obstructions/failure.png", cv2.IMREAD_UNCHANGED),
        cv2.imread("Obstructions/line.png", cv2.IMREAD_UNCHANGED),
        cv2.imread("Obstructions/split.png", cv2.IMREAD_UNCHANGED)
    ]

    #base image
    base = cv2.imread("Data/TestSet1/test1/expected.png", cv2.IMREAD_UNCHANGED)


    #iterate given number of times
    for _ in range(iters):
        #copy the base image
        modified_image = np.copy(base)

        #place random overlay on base image
        #NOTE: This routine is ran in-place, it does not create a copy itself.
        randomly_place_obstruction(modified_image, random.choice(obstructions))

        #now
        mask = create_modification_mask(base, modified_image)

        #display the result
        cv2.imshow("Modified Image", modified_image)
        cv2.imshow("Mask", mask)
        cv2.waitKey()





def test_the_whole_shebang(iters):
    # list of obstructed overlays
    obstructions = [
        cv2.imread("Obstructions/corner.png", cv2.IMREAD_UNCHANGED),
        cv2.imread("Obstructions/failure.png", cv2.IMREAD_UNCHANGED),
        cv2.imread("Obstructions/line.png", cv2.IMREAD_UNCHANGED),
        cv2.imread("Obstructions/split.png", cv2.IMREAD_UNCHANGED)
    ]

    # base image
    base = cv2.imread("Data/TestSet1/test1/expected.png", cv2.IMREAD_UNCHANGED)

    # iterate given number of times
    for _ in range(iters):
        # copy the base image
        modified_image = np.copy(base)

        #transform random obstruction
        transformed_obstruction = randomly_transform_obstruction(random.choice(obstructions))

        # place random overlay on base image
        # NOTE: This routine is ran in-place, it does not create a copy itself.
        randomly_place_obstruction(modified_image, transformed_obstruction)

        # now
        mask = create_modification_mask(base, modified_image)

        # display the result
        cv2.imshow("Modified Image", modified_image)
        cv2.imshow("Mask", mask)
        cv2.waitKey()








def main():
    test_the_whole_shebang(20)


if __name__ == "__main__":
    main()
