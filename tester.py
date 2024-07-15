import cv2 as cv
import numpy as np
import os
import sys
import cloudclassifyANN as cc

CLUSTERS = 15

EPOCHS = 15

CONF_THRESH = 0.6
SKY_THRESH = 0.1
NEG_THRESH = 0.1

NMS_THRESH = 0.3

TEST_LOCATION = '../../Data/TestPhotos/BackgroundTest/'
TEST_FILES = ['IFAK4250.JPG','TEST.JPG','test6.JPG', 'test11.JPG', 'test12.JPG']

cloud = cc.CloudClassify()
cloud.set_parameters(
    epochs = EPOCHS,
    conf_thresh = CONF_THRESH,
    sky_conf = SKY_THRESH,
    neg_conf = NEG_THRESH,
    nms_thresh = NMS_THRESH)
cloud.set_architecture(15, [64]) # output layer (num classes) not allowed to change
cloud.train()

for test in TEST_FILES:
    test_path = TEST_LOCATION + test
    cloud.run(test_path)
