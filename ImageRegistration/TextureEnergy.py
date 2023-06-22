"""
This file explores law's texture masks and kernels.

TODO: finish writing out the details of this file
"""


import numpy as np
import cv2
import matplotlib.pyplot as plt



#filter vectors
level = np.array([[1, 4, 6, 4, 1]])
edge = np.array([[-1, -2, 0, 2, 1]])
spot = np.array([[-1, 0, 2, 0, -1]])
ripple = np.array([[1, -4, 6, -4, 1]])

#edge kernels
el = np.dot(edge.reshape(-1, 1), level)
ee = np.dot(edge.reshape(-1, 1), edge)
es = np.dot(edge.reshape(-1, 1), spot)
er = np.dot(edge.reshape(-1, 1), ripple)

#level kernels
ll = np.dot(level.reshape(-1, 1), level)
le = np.dot(level.reshape(-1, 1), edge)
ls = np.dot(level.reshape(-1, 1), spot)
lr = np.dot(level.reshape(-1, 1), ripple)

#spot kernels
sl = np.dot(spot.reshape(-1, 1), level)
se = np.dot(spot.reshape(-1, 1), edge)
ss = np.dot(spot.reshape(-1, 1), spot)
sr = np.dot(spot.reshape(-1, 1), ripple)

#ripple kernels
rl = np.dot(ripple.reshape(-1, 1), level)
re = np.dot(ripple.reshape(-1, 1), edge)
rs = np.dot(ripple.reshape(-1, 1), spot)
rr = np.dot(ripple.reshape(-1, 1), ripple)

texture_kernels = [ # list of 2-tuples: ('name', kernel)
    ('edge x level', el), ('edge x edge', ee), ('edge x spot', es), ('edge x ripple', er),
    ('level x level', ll), ('level x edge', le), ('level x spot', ls), ('level x ripple', lr),
    ('spot x level', sl), ('spot x edge', se), ('spot x spot', ss), ('spot x ripple', sr),
    ('ripple x level', rl), ('ripple x edge', re), ('ripple x spot', rs), ('ripple x ripple', rr),
]



#shows an image
def show(image, title=None):
    plt.imshow(image, cmap='gray')
    if not (title is None):
        plt.title(title)
    plt.show()



#First Step In Law's procedure: Remove illumination by subtracting mean blur from original.
#He used a 15x15 window size and a mean blur.
#I might try different window sizes and a gaussian blur perhaps.
def remove_illumination(img, blur_mode='mean', window_size=15):
    #input validation
    assert blur_mode in ['mean', 'gauss'], "invalid blur mode, options are 'mean' and 'gauss'"
    assert (window_size % 2) > 0, 'window size must be odd'
    assert window_size > 2, 'window size must be 3 or more, in addition to being odd'


    #blur the image
    if blur_mode == 'mean':
        blur = cv2.blur(img, (window_size, window_size))
    else:
        blur = cv2.GaussianBlur(img, (window_size, window_size), 0)


    #subtract blur
    return img - blur



def test_removing_illumination():
    image = cv2.imread('images/fail4.jpg')
    out = remove_illumination(image)
    show(image, 'original')
    show(out, 'illumination removed')

    r1 = cv2.filter2D(out, ddepth=-1, kernel=el)
    r2 = cv2.filter2D(image, ddepth=-1, kernel=el)
    show(r1, 'mask with illumination removed')
    show(r2, 'mask without preprocessing')


#Applies the 16 kernels to the image to create 16 bands of texture features,
# (each band has 3 colors).
#Returns the 16 images as a list.
#Currently does not label the different bands in any significant way
def generate_texture_features(img):
    img = remove_illumination(img)
    for descriptor, kernel in texture_kernels:
        out = cv2.filter2D(img, ddepth=-1, kernel=kernel)
        #cv2.imwrite("TextureMaskExamples/" + descriptor + ".png", out)  #this line saves the masks
        show(out, descriptor)


if __name__ == '__main__':
    generate_texture_features(cv2.imread('images/fail3.jpg'))


