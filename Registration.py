"""
This file will have all the routines for doing registration stuff
"""
import cv2
import SimpleITK as sitk




def registration(static, moving, ):
    """
    This will be a big function that allows me to do different types of image registration.
    """










####################################################################################################
# Tests
####################################################################################################

def test_convert_to_sitk():
    """
    Tests converting a numpy image to a sitk image
    """
    image = cv2.imread("Data/Teapot/1.png", cv2.IMREAD_GRAYSCALE)
    cv2.imwrite('ERASEME-cv2.png', image)
    itk_image = sitk.GetImageFromArray(image, isVector=False) #isVector is referring to the pixels; whether they multivalued or not
    sitk.WriteImage(itk_image, 'ERASEME-sitk.png')



def understand_bspline_initialization():
    """
    I want to do a few different tests here just to see how bsplines
    are being initialized for itk. What is going on visually with the bspline initialization?
    """
    #read in image. Convert to itk format
    image = cv2.imread("Data/Teapot/1.png", cv2.IMREAD_GRAYSCALE)
    itk_image = sitk.GetImageFromArray(image, isVector=False)

    #Initialze bspline similarly to the examples I've seen
    """
    spline_terms = 3
    transformDomainMeshSize = [spline_terms] * itk_image.GetDimension()
    print(transformDomainMeshSize)
    initial_transform: sitk.BSplineTransform = sitk.BSplineTransformInitializer(itk_image, transformDomainMeshSize)
    print("fixed parameters:", initial_transform.GetFixedParameters())
    print(initial_transform.GetTransformDomainMeshSize())
    initial_transform.SetIdentity() # from docs, "...like setting all the transform parameters to zero in created parameter space"
    print("parameters:", initial_transform.GetParameters())
    print("dimensions:", len(initial_transform.GetParameters()))
    """


    #intialize it a differently, without BSplineTransformInitializer()
    dimension = 2
    spline_order = 3
    direction_matrix_row_major = [1.0, 0.0, 0.0, 1.0]  # identity, mesh is axis aligned
    origin = [-1.0, -1.0]
    domain_physical_dimensions = [2, 2]
    mesh_size = [4, 3]

    bspline = sitk.BSplineTransform(dimension, spline_order)
    bspline.SetTransformDomainOrigin(origin)
    bspline.SetTransformDomainDirection(direction_matrix_row_major)
    bspline.SetTransformDomainPhysicalDimensions(domain_physical_dimensions)
    bspline.SetTransformDomainMeshSize(mesh_size)
    print(len(bspline.GetParameters()) // 2)
    print(bspline.GetDimension())


def test():
    """entry point for running tests"""
    understand_bspline_initialization()

if __name__ == '__main__':
    test()


