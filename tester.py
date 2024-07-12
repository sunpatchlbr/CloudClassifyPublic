import cv2 as cv
import numpy as np
import os
import sys
import cloudclassify as cc

cloud = cc.CloudClassify()
cloud.train()
cloud.run()
