import cv2 as cv
import numpy as np
import os
import sys
import cloudclassifyANNColor as cc
import itertools

CLUSTERS = 23
COLOR_BINS = 36 # per color channel
HIDDEN_LAYERS = [87]

EPOCHS = 300
CONF_THRESH = 0.76
SKY_WINDOW = -0.11, 0.11
NEG_WINDOW = -0.11, 0.11

NMS_THRESH = 0.18

NUM_TESTS = 15
ANN_CLASSES = ['NEG','Sky','Cumulus','Cirrus','Stratus']
TEST_CLASSES = ['Cirrus', 'Cumulus', 'Stratus']
TEST_LOCATION = '../../Data/TestPhotos/TESTS/'
OUTPUT_LOCATION = '../../Data/Outputs/'

cloud = cc.CloudClassify()
cloud.set_parameters(
    epochs = EPOCHS,
    conf_thresh = CONF_THRESH,
    sky_window = SKY_WINDOW,
    neg_window = NEG_WINDOW,
    nms_thresh = NMS_THRESH)
cloud.set_architecture(CLUSTERS, COLOR_BINS, HIDDEN_LAYERS)
cloud.prepare()
 
def test_class(class_name):
    obstructed_accuracy = 0.0
    o_total = 0.0
    unobstructed_accuracy = 0.0
    u_total = 0.0
    for i in range(NUM_TESTS):
        print("Testing ", class_name, " ", i+1)
        u_file = "UNOBSTRUCTED/" + class_name + str(i+1) + ".JPG"
        o_file = "OBSTRUCTED/" + class_name + str(i+1) + ".JPG"
        u_path = TEST_LOCATION + u_file
        o_path = TEST_LOCATION + o_file
        #print("Testing ",u_path)
        u_output, u_predominant = cloud.run(u_path)
        if (ANN_CLASSES[u_predominant] == class_name):
            u_total += 1.0
        o_output, o_predominant = cloud.run(o_path)
        if (ANN_CLASSES[o_predominant] == class_name):
            o_total += 1.0
        cv.imwrite(OUTPUT_LOCATION+u_file, u_output)
        cv.imwrite(OUTPUT_LOCATION+o_file, o_output)
    obstructed_accuracy = o_total / NUM_TESTS
    unobstructed_accuracy = u_total / NUM_TESTS
    print("Obstructed accuracy: ", obstructed_accuracy)
    print("Unobstructed accuracy: ", unobstructed_accuracy)
    return (obstructed_accuracy, unobstructed_accuracy)

accuracies = []

for class_name in TEST_CLASSES:
    un, ob = test_class(class_name)
    accuracies.append([un, ob])

print()
print("Parameters: ")
print("BOW Clusters: ", CLUSTERS)
print("Bins per channel: ", COLOR_BINS)
print("Hidden layers: ", HIDDEN_LAYERS)
print("Epochs: ", 300)
print("Confidence threshold: ",  CONF_THRESH)
print("Sky tolerances: ", SKY_WINDOW)
print("Negative tolerances: ", NEG_WINDOW)
print("NMS Thresh: ", NMS_THRESH)
print()
print()

print("Final Accuracies: ")
print()
for accs, class_name in zip(accuracies, TEST_CLASSES):
    print(class_name, " unobstructed: ", accs[0])
    print(class_name, " obstructed: ", accs[1])
    print()
