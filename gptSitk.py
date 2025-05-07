import SimpleITK as sitk
import numpy as np
import cv2
import matplotlib.pyplot as plt
import Graphics
from AlignImages import PerspectiveRegistration







def first_test():
    # Read fixed and moving images
    fixed_image = sitk.ReadImage('Data/Teapot/ShapesTeapot.png', sitk.sitkFloat32)
    moving_image = sitk.ReadImage('Data/Teapot/4.png', sitk.sitkFloat32)
    #fixed_image = sitk.ReadImage('Data/Teapot/ShapesTeapot.png', sitk.sitkVectorFloat32)    #read as color image
    #moving_image = sitk.ReadImage('Data/Teapot/4.png', sitk.sitkVectorFloat32)

    #print("image depth:", moving_image.GetNumberOfComponentsPerPixel())

    # Initial transform using an affine transformation
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.AffineTransform(fixed_image.GetDimension()),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    # Set up the image registration method
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings
    registration_method.SetMetricAsMeanSquares()
    #registration_method.SetMetricAsMattesMutualInformation()

    # Optimizer settings
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Interpolator settings
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Set the initial transform
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Multi-resolution framework
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Perform the registration
    final_transform: sitk.SimpleITK.CompositeTransform = registration_method.Execute(fixed_image, moving_image)
    print(final_transform.GetParameters())
    print(final_transform)

    # Apply the final transform to the moving image
    resampled_image = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

    #show details of the transform
    #print(final_transform)
    #print(type(final_transform))

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

    # Print the final transform parameters
    print("Final transform parameters:")
    print(final_transform)

def sample1():
    """
    This is the sample from the ITK documentation that uses gradient descent.
    It can be found here:
        https://simpleitk.readthedocs.io/en/v1.1.0/Examples/ImageRegistrationMethod1/Documentation.html
    """
    fixed = sitk.ReadImage('Data/Teapot/ShapesTeapot.png', sitk.sitkFloat32)
    moving = sitk.ReadImage('Data/Teapot/4.png', sitk.sitkFloat32)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetOptimizerAsRegularStepGradientDescent(10.0, .01, 1000)
    R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
    R.SetInterpolator(sitk.sitkLinear)
    def command_iteration(method):
        print("{0:3} = {1:10.5f} : {2}".format(method.GetOptimizerIteration(), method.GetMetricValue(),method.GetOptimizerPosition()))
    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))
    outTx = R.Execute(fixed, moving)

    #the following code I added to visualize the results
    resampled_image = sitk.Resample(moving, fixed, outTx, sitk.sitkLinear, 0.0, moving.GetPixelID())
    fixed_array = sitk.GetArrayViewFromImage(fixed)
    resampled_array = sitk.GetArrayViewFromImage(resampled_image)
    fixed_array_cv = cv2.normalize(fixed_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    resampled_array_cv = cv2.normalize(resampled_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    composite = Graphics.BlendImages(resampled_array_cv, fixed_array_cv)
    cv2.imshow('composite', composite)
    cv2.waitKey(0)
    cv2.destroyAllWindows()













#global to hold metric values
metric_values = []

def step_callback(method):
    """
    print("{0:3} = {1:10.5f} : {2}".format(method.GetOptimizerIteration(),
                                   method.GetMetricValue(),
                                   method.GetOptimizerPosition()))
    """
    #print("iter:", method.GetOptimizerIteration())
    metric_values.append(method.GetMetricValue())
    i = method.GetOptimizerIteration()
    if i%10 == 0:
        print("step")



def test2():
    fixed_image = sitk.ReadImage('Data/Teapot/ShapesTeapot.png', sitk.sitkFloat32)
    moving_image = sitk.ReadImage('Data/Teapot/4.png', sitk.sitkFloat32)
    # fixed_image = sitk.ReadImage('Data/Teapot/ShapesTeapot.png', sitk.sitkVectorFloat32)    #read as color image
    # moving_image = sitk.ReadImage('Data/Teapot/4.png', sitk.sitkVectorFloat32)

    # print("image depth:", moving_image.GetNumberOfComponentsPerPixel())

    # Initial transform using an affine transformation
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.AffineTransform(fixed_image.GetDimension()),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    # Set up the image registration method
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings
    registration_method.SetMetricAsMeanSquares()
    # registration_method.SetMetricAsMattesMutualInformation()

    # Optimizer settings
    #registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=200)
    registration_method.SetOptimizerAsLBFGSB()
    #registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetOptimizerScalesFromJacobian()

    # Interpolator settings
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Set the initial transform
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Multi-resolution framework
    #registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    #registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])

    #using no shrinking or smoothing
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()


    #add iteration callback
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: step_callback(registration_method))


    # Perform the registration
    final_transform: sitk.SimpleITK.CompositeTransform = registration_method.Execute(fixed_image, moving_image)
    #print(final_transform.GetParameters())
    print(final_transform)
    print("inverse:", final_transform.GetInverse())
    print("iunverse params:", final_transform.GetInverse().GetParameters())

    print("number of transforms:", final_transform.GetNumberOfTransforms())
    print("first transform:", final_transform.GetNthTransform(0))
    #m11, m12, m21, m22, x, y =


    #print("metric value:", registration_method.GetMetricValue())

    plt.plot(metric_values)
    plt.title("LBFGS")
    plt.show()



    #now draw the transform
    M = np.array([   # first test
        [0.948813, 0.0160547, -0.486098],
        [-0.0189645, 0.891546, -4.94337]
    ])

    M = np.array([  # inverse?
        [1.05357, -0.0189724, -0.486098],
        [0.022411, 1.12124, -4.94337]
    ])


    M = np.array([  # inverse
        [1.05357, -0.0189724, -39.9663 - 0.486098],
        [0.022411, 1.12124, -71.7642 - 4.94337]
    ])

    M = np.array([
        [1.05357, -0.0189724, -39.9663 - 0.486098],
        [0.022411, 1.12124, -71.7642 - 4.94337]
    ])
    M = np.array([  # inverse
        [1.04582, -0.0129669, -36.9686],
        [0.0236557, 1.12059, -87.7544]
    ])


    fixed_image = cv2.imread('Data/Teapot/ShapesTeapot.png')


    image = cv2.imread('Data/Teapot/4.png')
    cv2.imshow("fixed", fixed_image)
    cv2.imshow("moving original", image)
    image = cv2.warpAffine(image, M, (fixed_image.shape[1], fixed_image.shape[0]))
    cv2.imshow("warped", image)

    blended = Graphics.BlendImages(image, fixed_image)
    cv2.imshow("blended", blended)

    cv2.waitKey()



def itk_to_cv(image):
    out = sitk.GetArrayFromImage(image)
    out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return out

def figure_out_image_representation():
    fixed = sitk.ReadImage('Data/Teapot/ShapesTeapot.png', sitk.sitkFloat32)
    print(fixed.GetPixelIDTypeAsString())
    moving = sitk.ReadImage('Data/Teapot/5.png')



    print(moving.GetPixelIDTypeAsString())
    print(moving.GetPixel((56, 40)))

    cv_moving = cv2.imread('Data/Teapot/5.png', cv2.IMREAD_GRAYSCALE)
    cv_moving = cv_moving.astype(np.float32)
    print(cv_moving[56, 40])
    image = sitk.GetImageFromArray(cv_moving, isVector=False)
    print(image.GetPixel((40, 56)))
    print(image.GetPixelIDTypeAsString())
    print('------------')
    print(cv_moving[238, 645])
    print(image.GetPixel((645, 238)))
    #print(cv_moving[40][56])
    #print(cv_moving.shape)
    #print(moving.GetSize())
    """
    print(moving.GetPixelIDTypeAsString())
    print(fixed.GetDepth())
    print(fixed.GetOrigin())
    print(fixed.GetSize())
    print(fixed.GetPixel((0, 0)))
    """



def just_numbers(transformation, similarity, optimizer, spline_terms=8, show_composite=False, maxiter=200):
    """
    --Similarity Metrics--
    --options example--
    options = {
        'transformation': 'bspline',  # affine, bspline
        'similarity': 'ms', # ms, mi, or ncc
        'optimizer': 'bfgs' # bfgs, ncg, gd, nm, or powell
    }
    """
    # ms, mi, or ncc
    options = {
        'transformation': 'bspline',  # affine, bspline
        'similarity': 'ms', # ms, mi, or ncc
        'optimizer': 'bfgs' # bfgs, ncg, gd, nm, or powell
    }

    #for reading a tiff

    """
    reader = sitk.ImageFileReader()
    reader.SetImageIO("TIFFImageIO")
    reader.SetFileName("Data/Teapot/ShapesTeapot-gray.tiff", sitk.sitkFloat32)
    fixed_image = reader.Execute()

    reader2 = sitk.ImageFileReader()
    reader2.SetImageIO("TIFFImageIO")
    reader2.SetFileName("Data/Teapot/5-gray.tiff", sitk.sitkFloat32)
    moving_image = reader.Execute()
    """




    #original images
    #fixed_image = sitk.ReadImage('Data/Teapot/ShapesTeapot.png', sitk.sitkFloat32)
    #moving_image = sitk.ReadImage('Data/Teapot/5.png', sitk.sitkFloat32)

    #some test images
    fixed_image = sitk.ReadImage('/Users/aidan/Desktop/ExpoSet/expected/15.png', sitk.sitkFloat32)
    moving_image = sitk.ReadImage('/Users/aidan/Desktop/ExpoSet/observed/15.png', sitk.sitkFloat32)



    #fixed_image = cv2.imread('Data/Teapot/ShapesTeapot.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
    #moving_image = cv2.imread('Data/Teapot/5.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
    #fixed_image = sitk.GetImageFromArray(fixed_image, isVector=False)
    #moving_image = sitk.GetImageFromArray(moving_image, isVector=False)
    #fixed_image = sitk.ReadImage('Data/Teapot/ShapesTeapot.png', sitk.sitkFloat32)
    #moving_image = sitk.ReadImage('Data/Teapot/5.png', sitk.sitkFloat32)



    #fixed_image = sitk.ReadImage('Data/Teapot/ShapesTeapot-gray.tiff', sitk.sitkFloat32)
    #moving_image = sitk.ReadImage('Data/Teapot/5-gray.tiff', sitk.sitkFloat32)


    #set transformation
    if transformation == 'bspline':
        transformDomainMeshSize = [spline_terms] * moving_image.GetDimension()
        initial_transform = sitk.BSplineTransformInitializer(fixed_image, transformDomainMeshSize)

    else: #transformation == 'affine'
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image,
            moving_image,
            sitk.AffineTransform(fixed_image.GetDimension()),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )



    # Instantiate registration method object
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings
    if similarity == 'ms':
        registration_method.SetMetricAsMeanSquares()
    elif similarity == 'mi':
        registration_method.SetMetricAsMattesMutualInformation()
    elif similarity == 'ncc':
        registration_method.SetMetricAsCorrelation()


    #optimizer
    if optimizer == 'bfgs':
        registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-4,
                       numberOfIterations=maxiter,
                       maximumNumberOfCorrections=5,
                       maximumNumberOfFunctionEvaluations=1000,
                       costFunctionConvergenceFactor=1e+7)
    elif optimizer == 'ncg':
        registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=1.0, numberOfIterations=maxiter)
    elif optimizer == 'gd':
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=maxiter) #line search vs no linesearch?
        #registration_method.SetOptimizerAsRegularStepGradientDescent(4.0, .01, 200)
        #registration_method.SetOptimizerAsGradientDescentLineSearch(learningRate=1.0, numberOfIterations=200, convergenceMinimumValue=1e-5, convergenceWindowSize=5)
    elif optimizer == 'nm': #nelder mead
        registration_method.SetOptimizerAsAmoeba(simplexDelta=1.0, numberOfIterations=maxiter)
    elif optimizer == 'powell':
        registration_method.SetOptimizerAsPowell(numberOfIterations=maxiter)


    #More setup
    registration_method.SetOptimizerScalesFromJacobian() #Not sure what this does
    registration_method.SetInterpolator(sitk.sitkLinear) #interpolator
    registration_method.SetInitialTransform(initial_transform, inPlace=True) # True or False?

    # using no shrinking or smoothing
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[0])
    #registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    #registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])

    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # add iteration callback. metric at each iter stored in metric_values from this
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: step_callback(registration_method))
    metric_values.clear() # reset the global variable for storing convergence data.

    #Perform the registration
    print('----Params----')
    print("Transformation:", transformation)
    print("Similarity Metric:", similarity)
    print("Optimizer:", optimizer)
    print('Starting...')
    final_transform: sitk.SimpleITK.CompositeTransform = registration_method.Execute(fixed_image, moving_image)
    print("MIN:", metric_values[-1])
    print('Done!')


    #create plot
    plt.plot(metric_values)
    #plt.ylim(left=-1, right=1)
    plt.ylim(top=0, bottom=-1.0)
    if len(metric_values) <= 100:
        plt.xlim(left=0, right=100)
    plt.title(f"{optimizer}, {similarity}, {transformation}, spline terms:{spline_terms}")
    plt.show()



    #show the composite
    if show_composite:
        resampled_image = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
        fixed_array = sitk.GetArrayViewFromImage(fixed_image)
        moving_array = sitk.GetArrayViewFromImage(moving_image)
        resampled_array = sitk.GetArrayViewFromImage(resampled_image)
        fixed_array_cv = cv2.normalize(fixed_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        moving_array_cv = cv2.normalize(moving_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        resampled_array_cv = cv2.normalize(resampled_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imshow('Fixed Image', fixed_array_cv)
        cv2.imshow('Moving Image', moving_array_cv)
        cv2.imshow('Resampled Image', resampled_array_cv)
        composite = Graphics.BlendImages(resampled_array_cv, fixed_array_cv)
        cv2.imshow('composite', composite)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




#sample1()
just_numbers(transformation='bspline', similarity='ncc', optimizer='ncg', spline_terms=4, show_composite=True, maxiter=200)
#figure_out_image_representation()






