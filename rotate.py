# -*- coding: utf-8 -*-
"""
Automatically detect rotation and line spacing of an image of text using
Radon transform

If image is rotated by the inverse of the output, the lines will be
horizontal (though they may be upside-down depending on the original image)

It doesn't work with black borders
"""

from __future__ import division, print_function
from skimage.transform import radon
from PIL import Image
from numpy import asarray, mean, array, blackman
import numpy
from numpy.fft import rfft
import matplotlib.pyplot as plt
from matplotlib.mlab import rms_flat
try:
    # More accurate peak finding from
    # https://gist.github.com/endolith/255291#file-parabolic-py
    from parabolic import parabolic

    def argmax(x):
        return parabolic(x, numpy.argmax(x))[0]
except ImportError:
    from numpy import argmax

filename = '2Drotate.png'

# Load file, converting to grayscale
I = asarray(Image.open(filename).convert('L'))
I = I - mean(I)  # Demean; make the brightness extend above and below zero

# Do the radon transform and display the result
sinogram = radon(I)

plt.gray()

# Find the RMS value of each row and find "busiest" rotation,
# where the transform is lined up perfectly with the alternating dark
# text and white lines
r = array([rms_flat(line) for line in sinogram.transpose()])
rotation = argmax(r)
print('Rotation: {:.2f} degrees'.format(90 - rotation))

import argparse 
import cv2
import numpy as np

img = cv2.imread('2Drotate.png', 0)
rows,cols = img.shape
M = cv2.getRotationMatrix2D((cols/2,rows/2),90 - rotation,1)
dst = cv2.warpAffine(img,M,(cols,rows))

plt.plot(121),plt.imshow(dst),plt.title('Output')
plt.savefig('hello.png')
plt.show()

