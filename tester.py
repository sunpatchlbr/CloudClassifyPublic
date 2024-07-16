import cv2 as cv
import numpy as np
import os
import sys
import cloudclassifyANN as cc

CLUSTERS = 12
HIDDEN_LAYERS = [75]

EPOCHS = 20

CONF_THRESH = 0.7
SKY_THRESH = 0.06
NEG_THRESH = 1

NMS_THRESH = 0.25

TEST_LOCATION = '../../Data/TestPhotos/BackgroundTest/'
OUTPUT_LOCATION = '../../Data/Outputs/'
TEST_FILES = ['IFAK4250.JPG','IMG_5938.JPG','TEST.JPG','test6.JPG', 'test11.JPG', 'test12.JPG']

cloud = cc.CloudClassify()
cloud.set_parameters(
    epochs = EPOCHS,
    conf_thresh = CONF_THRESH,
    sky_conf = SKY_THRESH,
    neg_conf = NEG_THRESH,
    nms_thresh = NMS_THRESH)
cloud.set_architecture(12, HIDDEN_LAYERS) # output layer (num classes) not allowed to change
cloud.train()

for test in TEST_FILES:
    test_path = TEST_LOCATION + test
    output = cloud.run(test_path,resize_it=False)
    cv.imwrite(OUTPUT_LOCATION+test, output)
