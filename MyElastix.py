
"""
THis script will test the Elastix Library:
"""
import cv2
import itk
import numpy as np

def test1():
    #read in images
    #fixed, moving = cv2.imread("Data/TestingDiffDiff/test1/expected.png"), cv2.imread("Data/TestingDiffDiff/test1/unobstructed-aligned.png")
    #fixed, moving = itk.imread("Data/TestingDiffDiff/test1/expected.png", itk.UC), itk.imread("Data/TestingDiffDiff/test1/unobstructed-aligned.png", itk.UC)
    fixed, moving = itk.imread("/Users/aidan/Desktop/ExpoSet/expected/107.png", itk.UC), itk.imread("/Users/aidan/Desktop/ExpoSet/observed/107.png", itk.UC)


    #parameter map setup
    parameter_object = itk.ParameterObject.New()
    resolutions = 3
    parameter_map_bspline = parameter_object.GetDefaultParameterMap("bspline", resolutions, 20.0)
    parameter_object.AddParameterMap(parameter_map_bspline)
    print("done setting up parameter map")

    #call the registration function
    result_image, result_transform_params = itk.elastix_registration_method(fixed, moving, parameter_object=parameter_object, log_to_console=False)
    print("RESULT THING")
    print(result_transform_params)

    itk.imwrite(result_image, "TRYING-NEW-THING-TEMPHERE.png")



def multi_channel_registration_test1():
    """
    test saving the parameters for a registration and repeating it on all three channels of the image.
    THis should produce a color version of the registered image.

    RESULTS:
    When I do this, something is wrong with the size
    """
    #setup the registration.
    fixed, moving = itk.imread("Data/TestingDiffDiff/test1/expected.png", itk.UC), itk.imread("Data/TestingDiffDiff/test1/unobstructed.png", itk.UC)
    parameter_object = itk.ParameterObject.New()
    parameter_map_bspline = parameter_object.GetDefaultParameterMap("bspline", numberOfResolutions=3, finalGridSpacingInPhysicalUnits=10.0)
    parameter_object.AddParameterMap(parameter_map_bspline)

    #Calculate transformation and apply grayscale registration
    result_image, result_transform_params = itk.elastix_registration_method(fixed, moving, parameter_object=parameter_object, log_to_console=False)

    #separate images into bgr channels
    moving = cv2.imread("Data/TestingDiffDiff/test1/unobstructed.png")
    moving_b, moving_g, moving_r = moving[:, :, 0], moving[:, :, 1], moving[:, :, 2]
    moving_b, moving_g, moving_r = itk.image_from_array(moving_b), itk.image_from_array(moving_g), itk.image_from_array(moving_r)


    #apply transformation to each channel
    #transformed_b = itk.transformix_filter(moving_b, parameter_object=result_transform_params)
    #transformed_g = itk.transformix_filter(moving_g, parameter_object=result_transform_params)
    #transformed_r = itk.transformix_filter(moving_r, parameter_object=result_transform_params)
    transformed_b = itk.transformix_filter(moving_b, result_transform_params)
    transformed_g = itk.transformix_filter(moving_g, result_transform_params)
    transformed_r = itk.transformix_filter(moving_r, result_transform_params)

    #convert back to numpy arrays
    b, g, r = itk.array_from_image(transformed_b), itk.array_from_image(transformed_g), itk.array_from_image(transformed_r)
    output_image = np.stack([b, g, r], -1)
    cv2.imshow("original", moving)
    cv2.imshow("output_image", output_image)
    cv2.waitKey()


def multi_channel_registration_test2():
    """
    See what happens when using fixed and moving of same size.

    RESULTS:
    THAT IS IT, MUST BE SAME SIZE
    """
    #first, check the size of the images read
    fixed, moving = cv2.imread("Data/TestingDiffDiff/test5/expected.png"), cv2.imread("Data/TestingDiffDiff/test5/unobstructed-aligned.png")
    print("fixed shape:", fixed.shape)
    print("moving shape:", moving.shape)

    #setup the registration.
    fixed, moving = itk.imread("Data/TestingDiffDiff/test5/expected.png", itk.UC), itk.imread("Data/TestingDiffDiff/test5/unobstructed-aligned.png", itk.UC)
    parameter_object = itk.ParameterObject.New()
    parameter_map_bspline = parameter_object.GetDefaultParameterMap("bspline", numberOfResolutions=3, finalGridSpacingInPhysicalUnits=10.0)
    parameter_object.AddParameterMap(parameter_map_bspline)

    #Calculate transformation and apply grayscale registration
    result_image, result_transform_params = itk.elastix_registration_method(fixed, moving, parameter_object=parameter_object, log_to_console=False)

    #separate images into bgr channels
    moving = cv2.imread("Data/TestingDiffDiff/test5/unobstructed-aligned.png")
    moving_b, moving_g, moving_r = moving[:, :, 0], moving[:, :, 1], moving[:, :, 2]
    moving_b, moving_g, moving_r = itk.image_from_array(moving_b), itk.image_from_array(moving_g), itk.image_from_array(moving_r)


    #apply transformation to each channel
    transformed_b = itk.transformix_filter(moving_b, result_transform_params)
    transformed_g = itk.transformix_filter(moving_g, result_transform_params)
    transformed_r = itk.transformix_filter(moving_r, result_transform_params)

    #convert back to numpy arrays
    b, g, r = itk.array_from_image(transformed_b), itk.array_from_image(transformed_g), itk.array_from_image(transformed_r)
    output_image = np.stack([b, g, r], -1)
    cv2.imshow("original", moving)
    cv2.imshow("output_image", output_image)
    cv2.waitKey()


def figure_out_proper_registration_parameters():
    """
    See what happens when using fixed and moving of same size.

    RESULTS:
    THAT IS IT, MUST BE SAME SIZE
    """
    #first, check the size of the images read
    fixed, moving = cv2.imread("Data/TestingDiffDiff/test5/expected.png"), cv2.imread("Data/TestingDiffDiff/test5/unobstructed-aligned.png")
    print("fixed shape:", fixed.shape)
    print("moving shape:", moving.shape)

    #setup the registration.
    fixed, moving = itk.imread("Data/TestingDiffDiff/test5/expected.png", itk.UC), itk.imread("Data/TestingDiffDiff/test5/unobstructed-aligned.png", itk.UC)
    parameter_object = itk.ParameterObject.New()
    parameter_map_bspline = parameter_object.GetDefaultParameterMap("bspline", numberOfResolutions=1, finalGridSpacingInPhysicalUnits=100)
    for parameter_name, parameter_values in parameter_map_bspline.items():
        print(f"{parameter_name}: {parameter_values}")
    parameter_object.AddParameterMap(parameter_map_bspline)

    #Calculate transformation and apply grayscale registration
    result_image, result_transform_params = itk.elastix_registration_method(fixed, moving, parameter_object=parameter_object, log_to_console=False)

    #separate images into bgr channels
    moving = cv2.imread("Data/TestingDiffDiff/test5/unobstructed-aligned.png")
    moving_b, moving_g, moving_r = moving[:, :, 0], moving[:, :, 1], moving[:, :, 2]
    moving_b, moving_g, moving_r = itk.image_from_array(moving_b), itk.image_from_array(moving_g), itk.image_from_array(moving_r)


    #apply transformation to each channel
    transformed_b = itk.transformix_filter(moving_b, result_transform_params)
    transformed_g = itk.transformix_filter(moving_g, result_transform_params)
    transformed_r = itk.transformix_filter(moving_r, result_transform_params)

    #convert back to numpy arrays
    b, g, r = itk.array_from_image(transformed_b), itk.array_from_image(transformed_g), itk.array_from_image(transformed_r)
    output_image = np.stack([b, g, r], -1)
    cv2.imshow("original", moving)
    cv2.imshow("output_image", output_image)
    cv2.waitKey()



#test1()
#multi_channel_registration_test2()
figure_out_proper_registration_parameters()


