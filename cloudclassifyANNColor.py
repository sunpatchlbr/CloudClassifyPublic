import cv2 as cv
import numpy as np
import os
import itertools
from non_max_suppression import non_max_suppression_fast as nms

DATA_PATH = '../../Data/TestPhotos/'
CLASSES = ['NEG','Sky','Cumulus','Cirrus','Stratus']
NUM_CLASSES = len(CLASSES)

TRAINING_SAMPLES = 'samples.npy'
TRAINING_LABELS = 'labels.npy'
VOCAB_PATH = 'cluster_vocab.npy'

BOW_NUM_TRAINING_SAMPLES_PER_CLASS = 70
ANN_NUM_TRAINING_SAMPLES_PER_CLASS = 70

FLANN_INDEX_KDTREE = 1

class CloudClassify(object):
    def __init__(self):
        print("hello")
        self._inputImage = None
        self._classifier = None
        self._sift = None
        self._flann = None

        self._BOW_CLUSTERS = NUM_CLASSES * 4
        self._COLOR_BINS = 5
        self._INPUT_LAYERS = self._BOW_CLUSTERS + self._COLOR_BINS*3
        self._vocab = None
        self._bow_kmeans_trainer = None
        self._bow_extractor = None
        
        self._ann = None

        #Default values
        self._EPOCHS = 20
        self._ANN_CONF_THRESHOLD = 0.3
        self._SKY_WINDOW = 0.03, 0.08
        self._NEG_WINDOW = 0.03, 0.08

        self._ANN_LAYERS = [self._BOW_CLUSTERS, 64, NUM_CLASSES] # input are bow descriptors, output are classes

        self._NMS_OVERLAP_THRESHOLD = 0.3
        
        self._sky = None
        self._output = None
        self._READY = False
        
    def run(self, inputFilePath=""):
        if not os.path.exists(inputFilePath):
            print("Couldn't find input image file")
        else:
            self._inputImage = cv.imread(inputFilePath)
            return self.detect_and_classify(self._inputImage, inputFilePath)

    def get_path_data(self, data_class, i):
        path = DATA_PATH + data_class + "/" + data_class + str(i) + "R.JPG"
        return path
                
    def prepare(self):
        self.initialize_classifiers()

    def set_parameters(self, epochs, conf_thresh, sky_window, neg_window, nms_thresh):
        self._EPOCHS = epochs
        self._ANN_CONF_THRESHOLD = conf_thresh
        self._SKY_WINDOW = sky_window
        self._NEG_WINDOW = neg_window
        self._NMS_OVERLAP_THRESHOLD = nms_thresh

    def set_architecture(self, clusters, color_bins, inner_layers):
        self._BOW_CLUSTERS = clusters
        self._COLOR_BINS = color_bins
        input_layers = int(self._BOW_CLUSTERS + self._COLOR_BINS*3)
        layers = [input_layers]
        layers.extend(inner_layers)
        layers.append(NUM_CLASSES)
        print(layers)
        self._ANN_LAYERS = layers
        

    def initialize_classifiers(self):
        if not os.path.isdir(DATA_PATH):
            print('data not found')
            exit(1)
        else:
            self._sift = cv.xfeatures2d.SIFT_create()
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=9)
            search_params = {}
            self._flann = cv.FlannBasedMatcher(index_params, search_params)
            self._bow_extractor = cv.BOWImgDescriptorExtractor(self._sift, self._flann)

            self._ann = cv.ml.ANN_MLP_create()

            if os.path.exists(VOCAB_PATH):
                print('Loading vocab...')
                self.load_vocab(VOCAB_PATH)
            else:
                print('Reclustering...')
                self.prepare_vocab()
                
            self.train()   
            self._READY = True
            print("CLASSIFIER READY")
            
    def load_vocab(self, path):
        self._vocab = np.load(path)
        self._bow_extractor.setVocabulary(self._vocab)

    def extract_bow_descriptors(self, img):
        features = self._sift.detect(img)
        return self._bow_extractor.compute(img, features)

    def add_sample_bow(self, path):
        #print("Sampling: ", path)
        gray = cv.imread(path, cv.IMREAD_GRAYSCALE)
        gray.astype('uint8')
        keypoints, descriptors = self._sift.detectAndCompute(gray, None)
        if descriptors is not None:
            self._bow_kmeans_trainer.add(descriptors)

    def prepare_vocab(self):
        print("Preparing vocab for BOW Extractor...")
        self._bow_kmeans_trainer = cv.BOWKMeansTrainer(self._BOW_CLUSTERS)
        
        for class_name in CLASSES:
            for i in range(BOW_NUM_TRAINING_SAMPLES_PER_CLASS):
                path = self.get_path_data(class_name, i+1)
                self.add_sample_bow(path)

        self._vocab = self._bow_kmeans_trainer.cluster()
        self._bow_extractor.setVocabulary(self._vocab)
        np.save(VOCAB_PATH,self._vocab)
        print("Saving vocab for later...")

    def return_hists(self,roi):
        bgr_planes = cv.split(roi)
        c = []
        blue = cv.calcHist(bgr_planes,
                           [0],
                           None,
                           [self._COLOR_BINS],
                           (0,256),
                           accumulate=False)
        green = cv.calcHist(bgr_planes,
                           [1],
                           None,
                           [self._COLOR_BINS],
                           (0,256),
                           accumulate=False)
        red = cv.calcHist(bgr_planes,
                           [2],
                           None,
                           [self._COLOR_BINS],
                           (0,256),
                           accumulate=False)
        c.extend(blue)
        c.extend(green)
        c.extend(red)
        c = np.array(c)
        c = c.flatten()
        #normal_c = (c-np.min(c))/(np.max(c)-np.min(c))
        length = np.linalg.norm(c)
        normal_c = c/length
        return normal_c

    def get_combined_input(self,roi,descriptors):
        colors = np.array(self.return_hists(roi))
        combined_sample = []
        combined_sample.extend(descriptors)
        combined_sample.extend(colors)
        return np.array(combined_sample,np.float32)

    def train(self):
        print('Training ANN on vocab...')
        self._ann.setLayerSizes(np.array(self._ANN_LAYERS))
        self._ann.setActivationFunction(cv.ml.ANN_MLP_SIGMOID_SYM, 0.6, 1.0)
        self._ann.setTrainMethod(cv.ml.ANN_MLP_BACKPROP, 0.07, 0.07)
        self._ann.setTermCriteria(
            (cv.TERM_CRITERIA_MAX_ITER | cv.TERM_CRITERIA_EPS, 100, 1.0))

        samples = []
        labels = []

        if os.path.exists(TRAINING_SAMPLES) and os.path.exists(TRAINING_LABELS):
            print("Loading existing records")
            samples = np.load(TRAINING_SAMPLES)
            labels = np.load(TRAINING_LABELS)
        else:
            print("Retaking descriptors for records...")
            for class_id in range(NUM_CLASSES):
                class_name = CLASSES[class_id]
                for i in range(ANN_NUM_TRAINING_SAMPLES_PER_CLASS):
                    path = self.get_path_data(class_name, i)
                    current = cv.imread(path)
                    descriptors = self.extract_bow_descriptors(current)
                    if descriptors is None:
                        continue
                    combined_sample = self.get_combined_input(current,descriptors[0])
                    samples.append(combined_sample)
                    labels.append([class_id])
            samples = np.array(samples,np.float32)
            labels = np.array(labels,np.float32)
            
            np.save(TRAINING_SAMPLES,samples)
            np.save(TRAINING_LABELS,labels)
            
            print("Records saved...")

        for e in range(self._EPOCHS):
            for sample, class_id in zip(samples, labels):
                sample = np.array(sample,np.float32)
                identity = np.array(np.zeros(NUM_CLASSES),np.float32)
                identity[int(class_id)] = 1.0
                data = cv.ml.TrainData_create(sample, cv.ml.COL_SAMPLE, identity)
                if self._ann.isTrained():
                    self._ann.train(
                        data,
                        cv.ml.ANN_MLP_UPDATE_WEIGHTS | cv.ml.ANN_MLP_NO_OUTPUT_SCALE)
                else:
                    self._ann.train(
                        data,
                        cv.ml.ANN_MLP_NO_INPUT_SCALE | cv.ml.ANN_MLP_NO_OUTPUT_SCALE)

        print("ANN ready")
    
    def detect_and_classify(self, img, inputPath):
        if self._READY:
            print("Detecting and Classifying ", inputPath)
            original_img = cv.imread(inputPath)
            pos_rects = []
            for resized in self.pyramid(img):
                scale = original_img.shape[0] / float(resized.shape[0])
                print("scale: ", resized.shape)
                for x, y, roi in self.sliding_window(resized):
                    descriptors = self.extract_bow_descriptors(roi)
                    if descriptors is None:
                        continue
                    combined_input = self.get_combined_input(roi,descriptors[0])
                    prediction = self._ann.predict(np.array([combined_input]))
                    class_id = int(prediction[0])
                    confidence = prediction[1][0][class_id]
                    sky_conf = prediction[1][0][1]
                    neg_conf = prediction[1][0][0]
                    if ( confidence > self._ANN_CONF_THRESHOLD
                         and sky_conf < self._SKY_WINDOW[1]
                         and sky_conf > self._SKY_WINDOW[0]
                         and neg_conf < self._NEG_WINDOW[1]
                         and neg_conf > self._NEG_WINDOW[0] ):
                        h, w, channels = roi.shape
                        pos_rects.append(
                            [int(x * scale),
                             int(y * scale),
                             int((x+w) * scale),
                             int((y+h) * scale),
                             confidence,
                             sky_conf,
                             neg_conf,
                             class_id])
            pos_rects = nms(np.array(pos_rects), self._NMS_OVERLAP_THRESHOLD)
            counts = np.zeros(NUM_CLASSES, dtype=float)
            for x0, y0, x1, y1, score, sky_conf, neg_conf, class_id in pos_rects:
                cv.rectangle(original_img, (int(x0), int(y0)), (int(x1), int(y1)),
                              (10, 220, 255), 4)
                text = CLASSES[int(class_id)] + ' ' \
                    + ('%.2f' % score) + ' ' + ('%.2f' % sky_conf) \
                    + ' ' + ('%.2f' % neg_conf)
                counts[int(class_id)] += (float(x1-x0)) #* score)
                cv.putText(original_img, text, (int(x0), int(y0) + 20),
                            cv.FONT_HERSHEY_SIMPLEX, 1, (10, 220, 255), 4)
            predominant_id = 0
            current_max = 0.0
            for i in range(NUM_CLASSES):
                if counts[i] > current_max:
                    current_max = counts[i]
                    predominant_id = i
            print("PREDOMINANT: ", CLASSES[predominant_id])
            print("COUNTS: ", counts)
            predominant_text = "PREDOMINANT: " + \
                               CLASSES[int(predominant_id)] + ": " + \
                               ('%.2f' % current_max)
            cv.putText(original_img, predominant_text, (20,120),
                       cv.FONT_HERSHEY_SIMPLEX, 2, (20, 255, 50), 4)
            return original_img, int(predominant_id)
        else:
            print("not trained")
            exit(1)
        

    def sliding_window(self, img, step=10, window_size=(75, 50)):
        img_h, img_w, channels = img.shape
        window_w, window_h = window_size
        for y in range(0, img_w, step):
            for x in range(0, img_h, step):
                roi = img[y:y+window_h, x:x+window_w]
                roi_h, roi_w, channels = roi.shape
                if roi_w == window_w and roi_h == window_h:
                    yield (x, y, roi)

    def pyramid(self, img, scale_factor=1.2, min_size=(200, 200),
                max_size=(700, 700)):
        h, w, channels = img.shape
        min_w, min_h = min_size
        max_w, max_h = max_size
        while w >= min_w and h >= min_h:
            if w <= max_w and h <= max_h:
                yield img
            w /= scale_factor
            h /= scale_factor
            img = cv.resize(img, (int(w), int(h)),
                             interpolation=cv.INTER_AREA)
