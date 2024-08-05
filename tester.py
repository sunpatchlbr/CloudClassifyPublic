import cv2 as cv
import numpy as np
import os
import sys
import cloudclassifyANNColor as cc
import itertools
import matplotlib.pyplot as plt
from sklearn import metrics

CLUSTERS = 21
COLOR_BINS = 28 # per color channel
HIDDEN_LAYERS = [75]

EPOCHS = 300
CONF_THRESH = 0.8
SKY_WINDOW = -0.12, 0.12
NEG_WINDOW = -0.12, 0.12

NMS_THRESH = 0.18

NUM_TESTS = 15
ANN_CLASSES = ['NEG','Sky','Cumulus','Cirrus','Stratus']
TEST_CLASSES = ['Cirrus','Cumulus','Stratus']
TEST_LOCATION = '../../Data/TestPhotos/TESTS/'
OUTPUT_LOCATION = '../../Data/Outputs/'

actual = []
predicted = []

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
    u_total = 0.05
    for i in range(NUM_TESTS):
        print("Testing ", class_name, " ", i+1)
        u_file = "UNOBSTRUCTED/" + class_name + str(i+1) + ".JPG"
        o_file = "OBSTRUCTED/" + class_name + str(i+1) + ".JPG"
        u_path = TEST_LOCATION + u_file
        o_path = TEST_LOCATION + o_file
        #print("Testing ",u_path)
        
        u_output, u_predominant = cloud.run(u_path)
        actual.append(class_name)
        predicted.append(ANN_CLASSES[u_predominant])
        if (ANN_CLASSES[u_predominant] == class_name):
            u_total += 1.0
            
        o_output, o_predominant = cloud.run(o_path)
        actual.append(class_name)
        predicted.append(ANN_CLASSES[o_predominant])
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

print("Final Sensitivites within class tests ( TP / ( TP + FN ) ): ")
print()
for accs, class_name in zip(accuracies, TEST_CLASSES):
    print(class_name, " unobstructed: ", accs[0])
    print(class_name, " obstructed: ", accs[1])
    print()

print()
print("Calculating Confusion Matrices and Metrics...")
print()

#calculate matrices
confusion_matrix = metrics.confusion_matrix(actual,
                                            predicted,
                                            labels=ANN_CLASSES)

individuals = metrics.multilabel_confusion_matrix(actual,
                                                  predicted,
                                                  labels=ANN_CLASSES)

# each class against each class
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix,
                                            display_labels=ANN_CLASSES)

def print_metrics(class_name, confusion_mat):
    [ [TN,FP],
      [FN, TP] ] = confusion_mat
    print(class_name)
    print()
    TOTAL = TN + FP + FN + TP
    accuracy = ( ( TP + TN ) / TOTAL )
    print("Accuracy ( ( TP + TN ) / TOTAL ): ", accuracy)
    precision = ( TP / ( TP + FP ) )
    print("Precision ( TP / ( TP + FP ) ): ", precision)
    sensitivity = ( TP / ( TP + FN ) )
    print("Sensitivity ( TP / ( TP + FN ) ): ", sensitivity)
    specificity = ( TN / ( TN + FP ) )
    print("Specificity ( TN / ( TN + FP ) ): ", specificity)
    print()
    print()
    return np.array([accuracy, precision, sensitivity, specificity], dtype=float)
    

print("Individual Metrics: ")
print()
# individual cms, normalized

mets = []

cirrus_cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=individuals[3],
                                            display_labels=['Other','Cirrus'])
mets.append(print_metrics('Cirrus',individuals[3]))

cumulus_cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=individuals[2],
                                            display_labels=['Other','Cumulus'])
mets.append(print_metrics('Cumulus',individuals[2]))

stratus_cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=individuals[4],
                                            display_labels=['Other','Stratus'])
mets.append(print_metrics('Stratus',individuals[4]))

mets = np.sum(mets, axis=0)
mets = np.divide(mets, 3.0)

# overall metrics
print("Overall metrics: ")
print("Average Accuracy: ", mets[0])
print("Average Precision: ", mets[1])
print("Average Sensitivity: ", mets[2])
print("Average Specificity: ", mets[3])


# plot
cm_display.plot()
cirrus_cm_display.plot()
cumulus_cm_display.plot()
stratus_cm_display.plot()

plt.show()
