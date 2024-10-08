"""
This script contains routines for calculating similarity between images.
There is a straightforward unit test towards the bottom that will run if this script is directly executed.

Here are the similarity metrics:
    mean squared difference - Images must have equal intensity distribution. Lower score is better.
    normalized correlation coefficient - Assumes linear relationship between intensities. 0 to 1, 1 is most similar.
    mutual information - More broadly applicable. 0 to 1, 1 is most similar.
    normalized mutual information - Can be even more broadly applicable.
    kappa statistic - Used for binary images only.


----QUOTE FROM ELASTIX DOCUMENTATION ON WHICH METRIC TO USE---
"The MSD measure is a measure that is only suited for two images with an equal intensity distribution,
i.e. for images from the same modality. NCC is less strict, it assumes a linear relation between the intensity
values of the fixed and moving image, and can therefore be used more often. The MI measure is even more
general: only a relation between the probability distributions of the intensities of the fixed and moving image
is assumed. For MI it is well-known that it is suited not only for mono-modal, but also for multi-modal
image pairs. This measure is often a good choice for image registration. The NMI measure is, just like
MI, suitable for mono- and multi-modality registration. Studholme et al. [1999] seems to indicate better
performance than MI in some cases. (Note that in elastix we optimized MI for performance, which we
did not do for NMI.) The KS measure is specifically meant to register binary images (segmentations). It
measures the “overlap” of the segmentations."
"""

import cv2
import numpy as np




########################################################################################################
# ----MEAN SQUARED DIFFERENCE----
# It is the sum of all the voxel distances squared (square inside summation), then divided by
# the number of voxels, hence the mean squared difference.
# Closer to zero means more similar.
########################################################################################################
def MeanSquaredDifference(moving, fixed):
    """
    Will resize the images if they are of a different shape.
    Uses the euclidean distance between voxels
    """
    #Handles different sizes
    different_sizes = moving.shape[0] == fixed.shape[0] and moving.shape[1] == fixed.shape[1]
    if not different_sizes:
        moving = cv2.resize(moving, (fixed.shape[1], fixed.shape[0]), interpolation=cv2.INTER_CUBIC)

    # convert to floats to avoid overload
    moving, fixed = moving.astype(np.float64), fixed.astype(np.float64)

    # core calculation
    squared_differences = (fixed - moving) ** 2
    msd = np.mean(squared_differences)
    return msd




########################################################################################################
# ----NORMALIZED CORRELATION COEFFICIENT----
# From 0 to 1, closer to 1 is more similar.
# Related to Pearson Correlation Coefficient.
########################################################################################################
def NormalizedCorrelationCoefficient(moving, fixed):
    #Check images are same shape/dimensions.
    assert moving.shape == fixed.shape, "Moving and fixed must have the same shape."

    # Calculate the mean of the fixed image and the moving image
    mean_fixed = np.mean(fixed)
    mean_moving = np.mean(moving)

    # Calculate the numerator of the NCC
    numerator = np.sum((fixed - mean_fixed) * (moving - mean_moving))

    # Calculate the denominator of the NCC
    denominator = np.sqrt(np.sum((fixed - mean_fixed) ** 2) * np.sum((moving - mean_moving) ** 2))

    # Calculate the NCC
    ncc = numerator / denominator
    return ncc






########################################################################################################
# ----MUTUAL INFORMATION----
# Definition from the following paper:
# Optimization of mutual information for multiresolution image registration,
# by P. Thevenaz and M. Unser, 2000, https://ieeexplore.ieee.org/document/887976.
# Closer to 1 means more similar.
########################################################################################################
def MutualInformation(moving, fixed, bins=256, normalized_subcall=False):
    """
    Calculate the Mutual Information (MI) between two images.

    Parameters:
    - moving: numpy array, the image to be compared.
    - fixed: numpy array, the reference image.
    - bins: int, number of bins to use for the histogram.
    - normalized_subcall: bool, indicates the function is used a subroutine in normalized_mutual_information,
        and will also return the probability values needed.

    Returns:
    - mi: float, the Mutual Information between the images.
    """
    # Check images are same shape/dimensions.
    assert moving.shape == fixed.shape, "Moving and fixed must have the same shape."

    # Compute the 2D histogram
    hist_2d, x_edges, y_edges = np.histogram2d(fixed.ravel(), moving.ravel(), bins=bins)

    # Convert bins counts to probability values
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)  # marginal for x (fixed_image)
    py = np.sum(pxy, axis=0)  # marginal for y (moving_image)

    # Compute the mutual information
    px_py = px[:, None] * py[None, :]  # broadcast to multiply marginals
    non_zero = pxy > 0  # only consider non-zero pxy values
    mi = np.sum(pxy[non_zero] * np.log2(pxy[non_zero] / px_py[non_zero]))

    #Return probabilities for a sub-call
    if normalized_subcall:
        return mi, pxy, px, py
    else:
        return mi





########################################################################################################
# ----NORMALIZED MUTUAL INFORMATION----
# Supposed to be a normalized and more robust version of Mutual Information by also using entropy.
########################################################################################################
def NormalizedMutualInformation(moving, fixed, bins=256):
    """
    Calculate the Normalized Mutual Information (NMI) between two images.

    Parameters:
    - moving: numpy array, the image to be compared.
    - fixed: numpy array, the reference image.
    - bins: int, number of bins to use for the histogram.

    Returns:
    - nmi: float, the Normalized Mutual Information between the images.
    """
    # Check images are same shape/dimensions.
    assert moving.shape == fixed.shape, "Moving and fixed must have the same shape."

    #Helper Routine for entropy
    def _entropy(prob):
        """Compute the entropy of a probability distribution."""
        prob = prob[prob > 0]  # Ignore zero probabilities to avoid log(0)
        return -np.sum(prob * np.log2(prob))

    #calculate mutual information
    mi, pxy, px, py = MutualInformation(moving, fixed, bins=bins, normalized_subcall=True)

    # Compute the entropies of the fixed and moving images
    h_fixed = _entropy(px)
    h_moving = _entropy(py)

    # Compute the normalized mutual information, that is to adjust for entropy
    #nmi = (h_fixed + h_moving) / mi
    nmi = 2 * mi / (h_fixed + h_moving)
    return nmi




########################################################################################################
# ----NORMALIZED MUTUAL INFORMATION Alt----
# This is the Chatgpt one that I can tell what the difference is.
# It gives the exact same results, EXCEPT for Data/TestSimilarity
########################################################################################################
def NormalizedMutualInformationAlt(fixed_image, moving_image, bins=256):
    # Helper Routine for entropy
    def _entropy(prob):
        """Compute the entropy of a probability distribution."""
        prob = prob[prob > 0]  # Ignore zero probabilities to avoid log(0)
        return -np.sum(prob * np.log2(prob))

    mi, pxy, px, py = MutualInformation(fixed_image, moving_image, bins=bins, normalized_subcall=False)
    h_fixed = _entropy(px)
    h_moving = _entropy(py)
    nmi = (2 * mi) / (h_fixed + h_moving)
    return nmi




########################################################################################################
# ----KAPPA STATISTIC----
# Meant for Binary images. Is essentially checking how well segments overlap.
########################################################################################################
def Kappa(moving, fixed, foreground_value=1):
    """
        Calculate the Kappa Statistic (KS) between two binary images.

        Parameters:
        - moving: numpy array, the image to be compared.
        - fixed: numpy array, the reference image.
        - foreground_value: int, the value of the foreground pixels (default is 1).

        Returns:
        - ks: float, the Kappa Statistic between the images.
        """
    # Check images are same shape/dimensions.
    assert moving.shape == fixed.shape, "Moving and fixed must have the same shape."

    # Indicator functions
    indicator_fixed = (fixed == foreground_value).astype(int)
    indicator_moving = (moving == foreground_value).astype(int)

    #Core kappa statistic calculation
    numerator = 2 * np.sum(indicator_fixed * indicator_moving) #numerator
    denominator = np.sum(indicator_fixed) + np.sum(indicator_moving) #denominator
    ks = numerator / denominator
    return ks




########################################################################################################
# ----UNIT TESTS----
# Runs through all the distance metrics
########################################################################################################
def unit_tests():
    print("Running unit tests for image similarity metrics...")

    #read in images
    observed, expected = cv2.imread("Data/TestingDiffDiff/test1/unobstructed-aligned.png"), cv2.imread("Data/TestingDiffDiff/test1/expected.png")
    penguin = cv2.resize(cv2.imread("Data/TestSimilarity/penguin.png"), (expected.shape[1], expected.shape[0]), interpolation=cv2.INTER_CUBIC)
    registered = cv2.imread("Data/TestSimilarity/itkFirstRegister.png", cv2.IMREAD_GRAYSCALE)
    expected_gray = cv2.cvtColor(expected, cv2.COLOR_BGR2GRAY)
    similar1, similar2 = cv2.imread("Data/TestSimilarity/similar1.png"), cv2.imread("Data/TestSimilarity/similar1.png") # similar images

    #Mean Squared Difference
    msd_penguin = MeanSquaredDifference(penguin, expected)
    msd_affine = MeanSquaredDifference(observed, expected)
    msd_spline = MeanSquaredDifference(registered, expected_gray)
    print("Mean Squared Difference, Penguin:", msd_penguin)
    print("Mean Squared Difference, Affine:", msd_affine)
    print("Mean Squared Difference, Spline:", msd_spline)

    #Normalized Correlation Coefficient
    ncc_affine = NormalizedCorrelationCoefficient(observed, expected)
    ncc_spline = NormalizedCorrelationCoefficient(registered, expected_gray)
    print("Normalized Correlation Coefficient, Affine:", ncc_affine)
    print("Normalized Correlation Coefficient, Spline:", ncc_spline)

    #Mutual Information
    mu_affine = MutualInformation(observed, expected)
    mu_spline = MutualInformation(registered, expected_gray)
    print("Mutual Information, Penguin:", MutualInformation(penguin, expected))
    print("Mutual Information, Affine:", mu_affine)
    print("Mutual Information, Spline:", mu_spline)
    print("Mutual Information, Similar:", MutualInformation(similar1, similar2))

    #Normalized Mutual Information
    nmu_affine = NormalizedMutualInformation(observed, expected)
    nmu_spline = NormalizedMutualInformation(registered, expected_gray)
    print("Normalized Mutual Information, Penguin:", NormalizedMutualInformation(penguin, expected))
    print("Normalized Mutual Information, Affine:", nmu_affine)
    print("Normalized Mutual Information, Spline:", nmu_spline)
    print("Normalized Mutual Information, Similar:", NormalizedMutualInformation(similar1, similar2))

    #Kappa statistic
    #_, otsu = cv2.threshold(obstructed_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    affine_edge = cv2.Canny(cv2.GaussianBlur(observed, (7, 7), 0), 100, 200)
    registered_edge = cv2.Canny(cv2.GaussianBlur(registered, (7, 7), 0), 100, 200)
    expected_edge = cv2.Canny(expected, 100, 200)
    similar1_edge, similar2_edge = cv2.Canny(similar1, 100, 200), cv2.Canny(similar2, 100, 200)
    penguin_edge = cv2.Canny(penguin, 100, 200)
    print("Kappa, Penguin+Edge:", Kappa(penguin_edge, expected_edge, foreground_value=255))
    print("Kappa, Affine+Edge:", Kappa(affine_edge, expected_edge, foreground_value=255))
    print("Kappa, Spline+Edge:", Kappa(registered_edge, expected_edge, foreground_value=255))
    print("Kappa, Similar+Edge:", Kappa(similar1_edge, similar2_edge, foreground_value=255))







################################################
# ENTRY POINT & MAIN
################################################
def main():
    unit_tests()

if __name__ == '__main__':
    main()
