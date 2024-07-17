import cv2 as cv
import numpy as np
import os
import sys
import cloudclassifyANN as cc

CLUSTERS = 17
HIDDEN_LAYERS = [75]

EPOCHS = 20
CONF_THRESH = 0.55
SKY_WINDOW = -0.02, 0.15
NEG_WINDOW = -0.01, 0.15

NMS_THRESH = 0.25

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
    output = cloud.run(test_path,resize_it=False)
    cv.imwrite(OUTPUT_LOCATION+test, output)
