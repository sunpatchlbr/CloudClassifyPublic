import cv2 as cv
import numpy as np
import os
import sys
import cloudclassifyANN as cc

CLUSTERS = 27
COLORS_BINS = 0
INPUT_LAYERS = CLUSTERS + COLORS_BINS
HIDDEN_LAYERS = [81]

EPOCHS = 120
CONF_THRESH = 0.75
SKY_WINDOW = -0.18, 0.18
NEG_WINDOW = -0.18, 0.18

NMS_THRESH = 0.3

TEST_LOCATION = '../../Data/TestPhotos/BackgroundTest/multi/'
OUTPUT_LOCATION = '../../Data/Outputs/'
TEST_FILES = ['TEST1.JPG',
              'TEST2.JPG',
              'TEST3.JPG',
              'TEST4.JPG',
              'TEST5.JPG',
              'TEST6.JPG',
              'TEST7.JPG',
              'TEST8.JPG',
              'TEST9.JPG',
              'TEST10.JPG',
              'TEST11.JPG',
              'TEST12.JPG',
              'TEST13.JPG',
              'TEST14.JPG']

cloud = cc.CloudClassify()
cloud.set_parameters(
    epochs = EPOCHS,
    conf_thresh = CONF_THRESH,
    sky_window = SKY_WINDOW,
    neg_window = NEG_WINDOW,
    nms_thresh = NMS_THRESH)
cloud.set_architecture(INPUT_LAYERS, HIDDEN_LAYERS)
cloud.prepare()

for test in TEST_FILES:
    test_path = TEST_LOCATION + test
    output = cloud.run(test_path)
    cv.imwrite(OUTPUT_LOCATION+test, output)
