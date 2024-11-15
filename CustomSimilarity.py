import SimpleITK as sitk
import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
import cv2





# Define the custom similarity metric (e.g., sum of absolute differences)
def custom_similarity_metric(transform_params):
    # Load fixed and moving images
    fixed_image = sitk.ReadImage("fixed_image.png", sitk.sitkFloat32)
    moving_image = sitk.ReadImage("moving_image.png", sitk.sitkFloat32)

    # Create a SimpleITK transform object using the parameters (e.g., translation)
    transform = sitk.TranslationTransform(2)  # Example: 2D translation
    transform.SetParameters(transform_params)

    # Resample the moving image using the transform
    resampled_moving_image = sitk.Resample(moving_image, fixed_image, transform)

    # Convert both images to numpy arrays for easier manipulation
    fixed_np = sitk.GetArrayFromImage(fixed_image)
    moving_np = sitk.GetArrayFromImage(resampled_moving_image)

    # Calculate the custom similarity metric (sum of absolute differences)
    similarity = np.sum(np.abs(fixed_np - moving_np))

    return similarity

def test1():

    # Initial guess for the transform parameters (e.g., translation x=0, y=0)
    initial_params = [0.0, 0.0]

    # Optimize the parameters using scipy.optimize.minimize
    result = minimize(custom_similarity_metric, initial_params, method='Powell')

    # Get the optimized transformation parameters
    optimized_params = result.x

    print("Optimized parameters:", optimized_params)











# convolution with a derivative of Gaussian kernel for image gradient
def normalize_to_uint8(image):
    """
    Normalize an image to the range [0, 255] and convert to uint8.
    """
    norm_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return norm_image.astype(np.uint8)
    #norm_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    #return norm_image





def compute_gradient(image, sigma=1.0):
    """
    Compute the intensity gradient of an image using the derivative of Gaussian.


    Parameters:
        image (2D numpy array): The input grayscale image.
        sigma (float): The standard deviation of the Gaussian.

    Returns:
        gradient_magnitude (2D numpy array): The gradient magnitude image.
        gradient_x (2D numpy array): The gradient in the x-direction.
        gradient_y (2D numpy array): The gradient in the y-direction.
    """
    # Apply Gaussian smoothing to the image
    smoothed_image = gaussian_filter(image, sigma=sigma)

    # Compute the gradient in x and y direction using the derivative of the Gaussian
    gradient_x = gaussian_filter(image, sigma=sigma, order=[0, 1])  # Derivative in x direction
    gradient_y = gaussian_filter(image, sigma=sigma, order=[1, 0])  # Derivative in y direction

    # Compute gradient magnitude
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    return gradient_magnitude, gradient_x, gradient_y


def test2():
    # Example usage:
    image = cv2.imread('Data/Teapot/4.png', cv2.IMREAD_GRAYSCALE)
    gradient_magnitude, gradient_x, gradient_y = compute_gradient(image, sigma=1.0)

    gradient_magnitude = gradient_magnitude / np.max(gradient_magnitude)
    gradient_x = gradient_x / np.max(np.abs(gradient_x))
    gradient_y = gradient_y / np.max(np.abs(gradient_y))
    print(gradient_magnitude.shape)
    print(gradient_magnitude.dtype)

    gradient_magnitude = normalize_to_uint8(gradient_magnitude.astype(np.float32))
    gradient_x = normalize_to_uint8(gradient_x.astype(np.float32))
    gradient_y = normalize_to_uint8(gradient_y.astype(np.float32))

    # Display the gradient magnitude using OpenCV (optional)
    cv2.imshow('Gradient Magnitude', gradient_magnitude)
    cv2.imshow('Gradient X', gradient_x)
    cv2.imshow('Gradient Y', gradient_y)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

test2()