import cv2 as cv
import numpy as np
import os    
from non_max_suppression import non_max_suppression_fast as nms

DATA_PATH = '../../Data/TestPhotos/'
CLASSES = ['Cumulus','NEG']
NUM_CLASSES = 2
TEST_PATH = '../../Data/TestPhotos/BackgroundTest/TEST.jpg'
BOW_CLUSTERS = 5
BOW_NUM_TRAINING_SAMPLES_PER_CLASS = 20
SVM_NUM_TRAINING_SAMPLES_PER_CLASS = 20

FLANN_INDEX_KDTREE = 1
SVM_SCORE_THRESHOLD = 1.8
NMS_OVERLAP_THRESHOLD = 0.1

class CloudClassify(object):
    def __init__(self):
        print("hello")
        self._inputImage = None
        self._resizedInput = None
        self._classifier = None
        self._sift = None
        self._flann = None
        self._bow_kmeans_trainer = None
        self._bow_extractor = None
        self._svm = None
        self._sky = None
        self._output = None
        self._READY = False
        
    def run(self, inputFileName=TEST_PATH):
        if not os.path.exists(inputFileName):
            print("Couldn't find input image file")
        else:
            self._inputImage = cv.imread(inputFileName)
            self.detect_and_classify(self._inputImage)
        
    def isolate_sky(self, originalImg, fg_proportion=0.4):
        print("Isolating sky from foreground...")

        cv.namedWindow("og", cv.WINDOW_NORMAL)
        cv.imshow("og", originalImg)

        mask = np.zeros(originalImg.shape[:2], np.uint8)

        height = originalImg.shape[0]

        fg_height = int(float(height) * fg_proportion)
        
        width = originalImg.shape[1]

        print("Height: ", height)
        print("Width: ", width)

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        rect = (0,height-fg_height, width, fg_height)
        
        cv.grabCut(originalImg, mask, rect, bgdModel, fgdModel, 15, cv.GC_INIT_WITH_RECT)

        obviousSkyMask = np.where((mask==2)|(mask==0), 1, 0).astype('uint8')

        obviousSky = originalImg*obviousSkyMask[:,:,np.newaxis]

        return obviousSky

    def get_path_data(self, data_class, i):
        path = DATA_PATH + data_class + "/" + data_class + str(i) + "R.JPG"
        return path
                
    def train(self):
        self.initialize_classifiers()

    def initialize_classifiers(self):
        if not os.path.isdir(DATA_PATH):
            print('data not found')
            exit(1)
        else:
            self._sift = cv.xfeatures2d.SIFT_create()
            
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=4)
            search_params = {}
            
            self._flann = cv.FlannBasedMatcher(index_params, search_params)

            self._bow_kmeans_trainer = cv.BOWKMeansTrainer(BOW_CLUSTERS)
            self._bow_extractor = cv.BOWImgDescriptorExtractor(self._sift, self._flann)

            for class_name in CLASSES:
                for i in range(BOW_NUM_TRAINING_SAMPLES_PER_CLASS):
                    path = self.get_path_data(class_name, i+1)
                    #print("PATH: ", path)
                    self.add_sample(path)

            voc = self._bow_kmeans_trainer.cluster()
            self._bow_extractor.setVocabulary(voc)

            training_data = []
            training_labels = []

            for c in range(NUM_CLASSES):
                class_name = CLASSES[c]
                for i in range(SVM_NUM_TRAINING_SAMPLES_PER_CLASS):
                    path = self.get_path_data(class_name, i+1)
                    dimg = cv.imread(path,cv.IMREAD_GRAYSCALE)
                    descriptors = self.extract_bow_descriptors(dimg)
                    if descriptors is not None:
                        training_data.extend(descriptors)
                        training_labels.append(c)

            self._svm = cv.ml.SVM_create()
            self._svm.setType(cv.ml.SVM_C_SVC)
            self._svm.setC(50)
            self._svm.setKernel(cv.ml.SVM_RBF)
            self._svm.train(np.array(training_data), cv.ml.ROW_SAMPLE, np.array(training_labels))
            self._READY = True
            print("SVM READY")
                

    def extract_bow_descriptors(self, img):
        features = self._sift.detect(img)
        return self._bow_extractor.compute(img, features)

    def add_sample(self, path):
        current = cv.imread(path, cv.IMREAD_GRAYSCALE)
        current.astype('uint8')
        keypoints, descriptors = self._sift.detectAndCompute(current, None)
        if descriptors is not None:
            self._bow_kmeans_trainer.add(descriptors)
    
    def detect_and_classify(self, img):
        if self._READY:
            print("Detecting and classifying clouds in sky...")
            gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            pos_rects = []
            for resized in self.pyramid(gray_img):
                print("resized: ", resized.shape)
                for x, y, roi in self.sliding_window(resized):
                    descriptors = self.extract_bow_descriptors(roi)
                    if descriptors is None:
                        continue
                    prediction = self._svm.predict(descriptors)
                    class_id = prediction[1][0][0]
                    #print("predict: ", prediction)
                    raw_prediction = self._svm.predict(
                        descriptors,
                        flags=cv.ml.STAT_MODEL_RAW_OUTPUT)
                    score = -raw_prediction[1][0][0]
                    #print("raw: ", raw_prediction)
                    if True: # class_id == 0.0: #or class_id != 2.0:
                        if score > SVM_SCORE_THRESHOLD:
                            print(score)
                            h, w = roi.shape
                            scale = gray_img.shape[0] / float(resized.shape[0])
                            pos_rects.append([int(x * scale),
                                              int(y * scale),
                                              int((x+w) * scale),
                                              int((y+h) * scale),
                                              score])
            pos_rects = nms(np.array(pos_rects), NMS_OVERLAP_THRESHOLD)
            #print('pos rects complete')
            print(pos_rects)
            for x0, y0, x1, y1, score in pos_rects:
                cv.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)),
                              (0, 255, 255), 2)
                text = ('%.2f' % score)
                cv.putText(img, text, (int(x0), int(y0) - 20),
                            cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv.imshow(TEST_PATH, img)
            cv.waitKey(0)
        else:
            print("not trained")
            exit(1)
        

    def sliding_window(self, img, step=75, window_size=(300, 200)):
        img_h, img_w = img.shape
        window_w, window_h = window_size
        for y in range(0, img_w, step):
            for x in range(0, img_h, step):
                roi = img[y:y+window_h, x:x+window_w]
                roi_h, roi_w = roi.shape
                if roi_w == window_w and roi_h == window_h:
                    yield (x, y, roi)

    def pyramid(self, img, scale_factor=2, min_size=(300, 500),
                max_size=(3000, 3000)):
        h, w = img.shape
        #print("current h:", h)
        #print("current w:", w)
        min_w, min_h = min_size
        max_w, max_h = max_size
        while w >= min_w and h >= min_h:
            if w <= max_w and h <= max_h:
                yield img
            w /= scale_factor
            h /= scale_factor
            img = cv.resize(img, (int(w), int(h)),
                             interpolation=cv.INTER_AREA)
