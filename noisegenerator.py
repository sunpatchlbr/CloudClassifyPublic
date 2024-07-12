import cv2 as cv
import numpy as np
import os  

NEG_DATA_PATH = '../../Data/Noise/'
NUM_TO_GENERATE = 20

for i in range(NUM_TO_GENERATE):
    randomByteArray = bytearray(os.urandom(180000))
    flatNumpyArray = np.array(randomByteArray).astype('uint8')
    grayImage = flatNumpyArray.reshape(200,300,3)
    cv.imwrite(NEG_DATA_PATH+"NEG"+str(i+1)+"R.JPG",grayImage)
