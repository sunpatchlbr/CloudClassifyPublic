import cv2 as cv
import numpy as np
import os
import sys
import cloudclassifyANNColor as cc

CLUSTERS = 24
COLOR_BINS = 8 # per color channel
HIDDEN_LAYERS = [70]

EPOCHS = 120
CONF_THRESH = 0.5
SKY_WINDOW = -0.2, 0.2
NEG_WINDOW = -0.2, 0.2

NMS_THRESH = 0.4

TEST_LOCATION = '../../Data/TestPhotos/BackgroundTest/multi/'
OUTPUT_LOCATION = '../../Data/Outputs/'
TEST_FILES = ['TEST18.JPG']

cloud = cc.CloudClassify()
cloud.set_parameters(
    epochs = EPOCHS,
    conf_thresh = CONF_THRESH,
    sky_window = SKY_WINDOW,
    neg_window = NEG_WINDOW,
    nms_thresh = NMS_THRESH)
cloud.set_architecture(CLUSTERS, COLOR_BINS, HIDDEN_LAYERS)
cloud.prepare()

for test in TEST_FILES:
    test_path = TEST_LOCATION + test
    output = cloud.run(test_path)
    cv.imwrite(OUTPUT_LOCATION+test, output)
