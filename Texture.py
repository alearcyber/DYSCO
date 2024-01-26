"""
TODO - Make a better description of this file.
This file has the texture stuff.
"""

import numpy as np
import cv2


####################################################################################################
# Constructing Texture Kernels
####################################################################################################

# filter vectors
level = np.array([[1, 4, 6, 4, 1]])
edge = np.array([[-1, -2, 0, 2, 1]])
spot = np.array([[-1, 0, 2, 0, -1]])
wave = np.array([[-1, 2, 0, -2, 1]])
ripple = np.array([[1, -4, 6, -4, 1]])

# edge kernels
el = np.dot(edge.reshape(-1, 1), level)
ee = np.dot(edge.reshape(-1, 1), edge)
es = np.dot(edge.reshape(-1, 1), spot)
ew = np.dot(edge.reshape(-1, 1), wave)
er = np.dot(edge.reshape(-1, 1), ripple)

# level kernels
ll = np.dot(level.reshape(-1, 1), level)
le = np.dot(level.reshape(-1, 1), edge)
ls = np.dot(level.reshape(-1, 1), spot)
lw = np.dot(level.reshape(-1, 1), wave)
lr = np.dot(level.reshape(-1, 1), ripple)

# spot kernels
sl = np.dot(spot.reshape(-1, 1), level)
se = np.dot(spot.reshape(-1, 1), edge)
ss = np.dot(spot.reshape(-1, 1), spot)
sw = np.dot(spot.reshape(-1, 1), wave)
sr = np.dot(spot.reshape(-1, 1), ripple)

# wave kernels
wl = np.dot(wave.reshape(-1, 1), level)
we = np.dot(wave.reshape(-1, 1), edge)
ws = np.dot(wave.reshape(-1, 1), spot)
ww = np.dot(wave.reshape(-1, 1), wave)
wr = np.dot(wave.reshape(-1, 1), ripple)

# ripple kernels
rl = np.dot(ripple.reshape(-1, 1), level)
re = np.dot(ripple.reshape(-1, 1), edge)
rs = np.dot(ripple.reshape(-1, 1), spot)
rw = np.dot(ripple.reshape(-1, 1), wave)
rr = np.dot(ripple.reshape(-1, 1), ripple)

epsilon = 1e-8
energy_maps = {
    'L5E5/E5L5': np.round(le / (el + epsilon), decimals=2),
    'L5R5/R5L5': lr // rl,
    'E5S5/S5E5': np.round(es / (se + epsilon), decimals=2),
    'S5S5': ss,
    'R5R5': rr,
    'L5S5/S5L5': np.round(ls / (sl + epsilon), decimals=2),
    'E5E5': ee,
    'E5R5/R5E5': np.round(er / (re + epsilon), decimals=2),
    'S5R5/R5S5': np.round(sr / (rs + epsilon), decimals=2)
}


def texture_features(img):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = np.empty(shape=(img.shape[0], img.shape[1], 0), dtype='uint8')
    for name in energy_maps:
        energy_map = energy_maps[name]
        layer = cv2.filter2D(img, ddepth=-1, kernel=energy_map)
        if len(img.shape) == 2: #for grayscale images
            layer = np.expand_dims(layer, axis=2)
        out = np.concatenate((out, layer), axis=2)
    return out
