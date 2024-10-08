import SimpleITK as sitk
import cv2
import numpy as np


def test_affine():
    moving_image = sitk.ReadImage("Data/Teapot/4.png", sitk.sitkFloat32)
    fixed_image = sitk.ReadImage("Data/Teapot/ShapesTeapot.png", sitk.sitkFloat32)

    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.AffineTransform(fixed_image.GetDimension()),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsJointHistogramMutualInformation()
    registration_method.SetOptimizerAsGradientDescentLineSearch(learningRate=1.0, numberOfIterations=50,
                                                                convergenceMinimumValue=1e-4, convergenceWindowSize=5)


    # Interpolator settings
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Set the initial transform
    registration_method.SetInitialTransform(initial_transform, inPlace=False)





    #YHER
    # Perform the registration
    final_transform = registration_method.Execute(fixed_image, moving_image)

    # Apply the final transform to the moving image
    resampled_image = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

    # Convert images to numpy arrays for visualization with OpenCV
    fixed_array = sitk.GetArrayViewFromImage(fixed_image)
    moving_array = sitk.GetArrayViewFromImage(moving_image)
    resampled_array = sitk.GetArrayViewFromImage(resampled_image)

    # Normalize the arrays to 8-bit images for OpenCV display (0-255 range)
    fixed_array_cv = cv2.normalize(fixed_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    moving_array_cv = cv2.normalize(moving_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    resampled_array_cv = cv2.normalize(resampled_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Display the images using OpenCV
    cv2.imshow('Fixed Image', fixed_array_cv)
    cv2.imshow('Moving Image', moving_array_cv)
    cv2.imshow('Resampled Image', resampled_array_cv)

    # Wait for a key press and close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

test_affine()
