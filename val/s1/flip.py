import cv2
import numpy

for i in range(101, 130):
    image = cv2.imread('file_{}.jpg'.format(i))
    flip_image = cv2.flip(image,1)
    cv2.imwrite('fil_{}.jpg'.format(i), flip_image)
