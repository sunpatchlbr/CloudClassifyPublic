import cv2 as cv
import numpy as np
import os
import sys
import cloudclassifyANN as cc

CLUSTERS = 14
HIDDEN_LAYERS = [72]

EPOCHS = 100
CONF_THRESH = 0.79
SKY_WINDOW = -0.1, 0.06
NEG_WINDOW = -0.06, 0.1

NMS_THRESH = 0.1

TEST_LOCATION = '../../Data/TestPhotos/BackgroundTest/multi/'
OUTPUT_LOCATION = '../../Data/Outputs/'
TEST_FILES = ['multi3.JPG','multi4.JPG']

cloud = cc.CloudClassify()
cloud.set_parameters(
    epochs = EPOCHS,
    conf_thresh = CONF_THRESH,
    sky_window = SKY_WINDOW,
    neg_window = NEG_WINDOW,
    nms_thresh = NMS_THRESH)
cloud.set_architecture(CLUSTERS, HIDDEN_LAYERS)
cloud.prepare()

for test in TEST_FILES:
    test_path = TEST_LOCATION + test
    output = cloud.run(test_path)
    cv.imwrite(OUTPUT_LOCATION+test, output)
