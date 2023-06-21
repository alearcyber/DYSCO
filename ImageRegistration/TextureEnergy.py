##
# level
#
#
#
#
#
#
#


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
#TODO FINISH BUILDINg the kernSL



#currently not used
def build_kernels():
    filters = [
        {'name': 'level', 'vector': np.array([[1, 4, 6, 4, 1]])},
        {'name': 'edge', 'vector': np.array([[-1, -2, 0, 2, 1]])},
        {'name': 'spot', 'vector': np.array([-1, 0, 2, 0, -1])},
        {'name': 'ripple', 'vector': np.array([1, -4, 6, -4, 1])}
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
    show(r1, 'mask')






if __name__ == '__main__':


    test_removing_illumination()


