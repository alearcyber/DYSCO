import SimpleITK as sitk


image = sitk.ReadImage("/Users/aidanlear/Desktop/teapot1-cropped.png")

print("done")
R = sitk.ImageRegistrationMethod()
R.SetMetricAsMattesMutualInformation(50)
R.SetOptimizerAsGradientDescentLineSearch(5.0, 100, convergenceMinimumValue=1e-4, convergenceWindowSize=5)
R.SetOptimizerAsGradientDescentLineSearch


