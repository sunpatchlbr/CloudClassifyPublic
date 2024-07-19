import cv2 as cv
import numpy as np
import os
import sys
import cloudclassifyANN as cc

CLUSTERS = 13
HIDDEN_LAYERS = [70]

EPOCHS = 100
CONF_THRESH = 0.75
SKY_WINDOW = -0.1,0.1#-0.08, 0.07
NEG_WINDOW = -0.1,0.1#-0.05, 0.08

NMS_THRESH = 0.1

TEST_LOCATION = '../../Data/TestPhotos/BackgroundTest/multi/'
OUTPUT_LOCATION = '../../Data/Outputs/'
TEST_FILES = ['multi2.JPG','multi3.JPG','multi4.JPG']

cloud = cc.CloudClassify()
cloud.set_parameters(
    epochs = EPOCHS,
    conf_thresh = CONF_THRESH,
    sky_window = SKY_WINDOW,
    neg_window = NEG_WINDOW,
    nms_thresh = NMS_THRESH)
cloud.set_architecture(CLUSTERS, HIDDEN_LAYERS) # output layer (num classes) not allowed to change
cloud.train()

for test in TEST_FILES:
    test_path = TEST_LOCATION + test
    output = cloud.run(test_path,remove_foreground=0.0)
    cv.imwrite(OUTPUT_LOCATION+test, output)
