"""
I will just be messing around with wavelets in this script
"""
import pywt
import numpy as np
import cv2


"""
Just does one level
"""
def haar_wvlt(image):
    # Ensure the image is in float format
    image = np.array(image, dtype=float)
    N, M = image.shape

    # Check if the image is square and dimensions are a power of 2
    assert N == M and ((N & (N - 1)) == 0), "Image must be square and dimensions must be a power of 2."

    # Initialize the transformed image
    transformed = np.zeros_like(image)

    # Horizontal processing
    for y in range(N):
        for x in range(0, N, 2):
            transformed[y, x // 2] = (image[y, x] + image[y, x + 1]) / 2  # Low-pass filter
            transformed[y, N // 2 + x // 2] = (image[y, x] - image[y, x + 1]) / 2  # High-pass filter

    # Vertical processing on the result from horizontal processing
    result = np.zeros_like(transformed)
    for x in range(N):
        for y in range(0, N, 2):
            result[y // 2, x] = (transformed[y, x] + transformed[y + 1, x]) / 2  # Low-pass filter
            result[N // 2 + y // 2, x] = (transformed[y, x] - transformed[y + 1, x]) / 2  # High-pass filter

    return result




""" this one does multi level"""
def haar_wavelet_transform_recursive(image, levels):
    image = np.array(image, dtype=np.double)
    def transform_single_level(image):
        N, M = image.shape
        transformed = np.zeros_like(image)
        # Horizontal processing
        for y in range(N):
            for x in range(0, N, 2):
                transformed[y, x // 2] = (image[y, x] + image[y, x + 1]) / 2 # Low-pass
                transformed[y, N // 2 + x // 2] = (image[y, x] - image[y, x + 1]) / 2 # High-pass
        # Vertical processing
        result = np.zeros_like(transformed)
        for x in range(N):
            for y in range(0, N, 2):
                result[y // 2, x] = (transformed[y, x] + transformed[y + 1, x]) / 2 # Low-pass
                result[N // 2 + y // 2, x] = (transformed[y, x] - transformed[y + 1, x]) / 2 # High-pass
        return result

    # Check if levels is positive
    if levels <= 0:
        return image

    # Transform the current level
    current_transform = transform_single_level(image)

    # The size of the next LL band is half the current image size
    N = current_transform.shape[0] // 2

    # Recursively apply the transform to the LL band if more levels are required
    if levels > 1:
        ll_band = current_transform[0:N, 0:N]
        current_transform[0:N, 0:N] = haar_wavelet_transform_recursive(ll_band, levels - 1)

    return current_transform


"""inverse for multi level"""
def inverse_haar_wavelet_transform(transformed, levels):
    def inverse_transform_single_level(image):
        N, M = image.shape
        result = np.zeros_like(image)
        # Vertical processing
        for x in range(N // 2):
            for y in range(0, N, 2):
                result[y, x] = image[y // 2, x] + image[N // 2 + y // 2, x] # Low-pass
                result[y + 1, x] = image[y // 2, x] - image[N // 2 + y // 2, x] # High-pass
        # Horizontal processing
        final_result = np.zeros_like(result)
        for y in range(N):
            for x in range(0, N, 2):
                final_result[y, x] = result[y, x // 2] + result[y, N // 2 + x // 2] # Low-pass
                final_result[y, x + 1] = result[y, x // 2] - result[y, N // 2 + x // 2] # High-pass
        return final_result

    current_image = transformed

    for _ in range(levels):
        N = current_image.shape[0] // (2 ** levels)
        current_level_image = np.zeros((N*2, N*2))
        current_level_image[:N, :N] = current_image[:N, :N]
        current_image[:N*2, :N*2] = inverse_transform_single_level(current_level_image)
        levels -= 1

    return current_image








def one():
    image = cv2.imread("Data/TestSet1/test1/cam-low-exposure.png", cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_CUBIC)
    out = haar_wvlt(image)
    cv2.imshow("original", image)
    cv2.imshow("wavy", out)
    cv2.waitKey()



def two():
    levels = 3
    image = cv2.imread("Data/TestSet1/test1/cam-low-exposure.png", cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_CUBIC)
    out = haar_wavelet_transform_recursive(image, levels)
    wadda = np.copy(out)
    restored = inverse_haar_wavelet_transform(wadda, levels)
    cv2.imshow("original", image)
    cv2.imshow("wavy", out)
    cv2.imshow("restored", restored)
    cv2.waitKey()




def using_library():
    image = cv2.imread("Data/TestSet1/test1/cam-low-exposure.png", cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_CUBIC)

    #coeffs = pywt.dwt2(image, 'db5')
    #cA, (cH, cV, cD) = coeffs

    c = pywt.wavedec2(image, 'db5', 2)

    #parse information
    cA = c[0]
    (cH1, cV1, cD1) = c[-1]
    (cH2, cV2, cD2) = c[-2]


    #approximate
    cA = c[0]
    cv2.imshow("approximate", cA)
    cv2.waitKey()
    cv2.destroyAllWindows()

    #level 1
    cv2.imshow("Horizontal", cH1)
    cv2.imshow("Vertical", cV1)
    cv2.imshow("Diagonal", cD1)
    cv2.waitKey()

    #level 2
    cv2.imshow("Horizontal", cH2)
    cv2.imshow("Vertical", cV2)
    cv2.imshow("Diagonal", cD2)
    cv2.waitKey()
    cv2.destroyAllWindows()




def main():
    using_library()

if __name__ == "__main__":
    main()