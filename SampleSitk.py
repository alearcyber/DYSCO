from __future__ import print_function
import SimpleITK as sitk
import sys
import os
import cv2
import numpy as np


def command_iteration(method):
    print("{0:3} = {1:10.5f} : {2}".format(method.GetOptimizerIteration(),
                                   method.GetMetricValue(),
                                   method.GetOptimizerPosition()))

"""
if len(sys.argv) < 4:
    print("Usage: {0} <fixedImageFilter> <movingImageFile> <outputTransformFile>".format(sys.argv[0]))
    sys.exit(1)
"""


arg1 = "Data/Teapot/ShapesTeapot.png" #fixed
arg2 = "Data/Teapot/4.png" #moving
arg3 = "TEMP.txt"



fixed = sitk.ReadImage(arg1, sitk.sitkFloat32)

moving = sitk.ReadImage(arg2, sitk.sitkFloat32)

R = sitk.ImageRegistrationMethod()
R.SetMetricAsMeanSquares()
R.SetOptimizerAsRegularStepGradientDescent(4.0, .01, 200)
#R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
R.SetInitialTransform(sitk.AffineTransform(fixed.GetDimension()))
R.SetInterpolator(sitk.sitkLinear)

R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

outTx = R.Execute(fixed, moving)

print("-------")
print(outTx)
print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
print(" Iteration: {0}".format(R.GetOptimizerIteration()))
print(" Metric value: {0}".format(R.GetMetricValue()))

#sitk.WriteTransform(outTx,  arg3)

if ( not "SITK_NOSHOW" in os.environ ):

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(outTx)

    out = resampler.Execute(moving)
    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    cimg = sitk.Compose(simg1, simg2, simg1//2.+simg2//2.)
    # sitk.Show(cimg, "ImageRegistration1 Composition")

    #I added the following 4 lines for visualizing
    resampled_array = sitk.GetArrayViewFromImage(cimg)
    resampled_array_cv = cv2.normalize(resampled_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imshow("", resampled_array_cv)
    cv2.waitKey()
