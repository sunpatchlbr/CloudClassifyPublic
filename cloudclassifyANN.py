import cv2 as cv
import numpy as np
import os    
from non_max_suppression import non_max_suppression_fast as nms

DATA_PATH = '../../Data/TestPhotos/'
CLASSES = ['NEG','Sky','Cumulus']
NUM_CLASSES = 3
TEST_PATH = '../../Data/TestPhotos/BackgroundTest/TEST.jpg'

MAX_HEIGHT = 1100

BOW_NUM_TRAINING_SAMPLES_PER_CLASS = 20
ANN_NUM_TRAINING_SAMPLES_PER_CLASS = 20

FLANN_INDEX_KDTREE = 1

class CloudClassify(object):
    def __init__(self):
        print("hello")
        self._inputImage = None
        self._resizedInput = None
        self._classifier = None
        self._sift = None
        self._flann = None

        self._BOW_CLUSTERS = NUM_CLASSES * 4
        self._bow_kmeans_trainer = None
        self._bow_extractor = None
        
        self._ann = None
        
        self._EPOCHS = 20
        self._ANN_CONF_THRESHOLD = 0.7
        self._SKY_THRESH = 0.07
        self._NEG_THRESH = 0.05

        self._ANN_LAYERS = [self._BOW_CLUSTERS, 64, NUM_CLASSES] # input are bow descriptors, output are classes

        self._NMS_OVERLAP_THRESHOLD = 0.3
        
        self._sky = None
        self._output = None
        self._READY = False
        
    def run(self, inputFileName=TEST_PATH):
        if not os.path.exists(inputFileName):
            print("Couldn't find input image file")
        else:
            self._inputImage = cv.imread(inputFileName)
            self._inputImage = self.resizeInput(self._inputImage, MAX_HEIGHT)
            self.detect_and_classify(self._inputImage)

    def get_path_data(self, data_class, i):
        path = DATA_PATH + data_class + "/" + data_class + str(i) + "R.JPG"
        return path
                
    def train(self):
        self.initialize_classifiers()

    def set_parameters(self, epochs, conf_thresh, sky_conf, neg_conf, nms_thresh):
        self._EPOCHS = 20
        self._ANN_CONF_THRESHOLD = 0.7
        self._SKY_THRESH = 0.07
        self._NEG_THRESH = 0.05

        self._NMS_OVERLAP_THRESHOLD = 0.3

    def set_architecture(self, clusters, inner_layers):
        self._BOW_CLUSTERS = clusters
        inner_layers.append(NUM_CLASSES)
        self._ANN_LAYERS = (np.array(clusters,int)).extend(inner_layers)
        

    def record(self, sample, classification):
        return (np.array([sample], np.float32),
                np.array([classification], np.float32))

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
                    self.add_sample(path)

            voc = self._bow_kmeans_trainer.cluster()
            self._bow_extractor.setVocabulary(voc)

            self._ann = cv.ml.ANN_MLP_create()
            self._ann.setLayerSizes(np.array(ANN_LAYERS))
            self._ann.setActivationFunction(cv.ml.ANN_MLP_SIGMOID_SYM, 0.6, 1.0)
            self._ann.setTrainMethod(cv.ml.ANN_MLP_BACKPROP, 0.1, 0.1)
            self._ann.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER | cv.TERM_CRITERIA_EPS, 100, 1.0))

            records = []

            for c in range(NUM_CLASSES):
                class_name = CLASSES[c]
                for i in range(ANN_NUM_TRAINING_SAMPLES_PER_CLASS):
                    current = cv.imread(self.get_path_data(class_name, i))
                    descriptors = self.extract_bow_descriptors(current)
                    if descriptors is None:
                        continue
                    sample = descriptors[0]
                    identity = np.zeros(NUM_CLASSES)
                    identity[c] = 1.0
                    record = self.record(sample, identity)
                    records.append(record)

            for e in range(EPOCHS):
                print("epoch: %d" % e)
                for t, c in records:
                    data = cv.ml.TrainData_create(t, cv.ml.ROW_SAMPLE, c)
                    if self._ann.isTrained():
                        self._ann.train(data, cv.ml.ANN_MLP_UPDATE_WEIGHTS | cv.ml.ANN_MLP_NO_OUTPUT_SCALE)
                    else:
                        self._ann.train(data, cv.ml.ANN_MLP_NO_INPUT_SCALE | cv.ml.ANN_MLP_NO_OUTPUT_SCALE)
            self._READY = True
            print("ANN READY")
                

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
                    prediction = self._ann.predict(descriptors)
                    class_id = int(prediction[0])
                    confidence = prediction[1][0][class_id]
                    sky_conf = abs(prediction[1][0][1])
                    NEG_conf = abs(prediction[1][0][0])
                    if confidence > ANN_CONF_THRESHOLD and sky_conf < SKY_THRESH and NEG_conf < NEG_THRESH:
                        h, w = roi.shape
                        scale = gray_img.shape[0] / \
                            float(resized.shape[0])
                        pos_rects.append(
                            [int(x * scale),
                             int(y * scale),
                             int((x+w) * scale),
                             int((y+h) * scale),
                             confidence,
                             sky_conf,
                             NEG_conf,
                             class_id])
            pos_rects = nms(np.array(pos_rects), NMS_OVERLAP_THRESHOLD)
            print(pos_rects)
            for x0, y0, x1, y1, score, sky_conf, NEG_conf, class_id in pos_rects:
                cv.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)),
                              (0, 255, 255), 2)
                text = CLASSES[int(class_id)] + ' ' \
                    + ('%.2f' % score) + ' ' + ('%.2f' % sky_conf) \
                    + ' ' + ('%.2f' % NEG_conf)
                cv.putText(img, text, (int(x0), int(y0) - 20),
                            cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv.imshow(TEST_PATH, img)
            cv.waitKey(0)
        else:
            print("not trained")
            exit(1)
        

    def sliding_window(self, img, step=40, window_size=(150, 100)):
        img_h, img_w = img.shape
        window_w, window_h = window_size
        for y in range(0, img_w, step):
            for x in range(0, img_h, step):
                roi = img[y:y+window_h, x:x+window_w]
                roi_h, roi_w = roi.shape
                if roi_w == window_w and roi_h == window_h:
                    yield (x, y, roi)

    def pyramid(self, img, scale_factor=1.5, min_size=(300, 300),
                max_size=(2000, 2000)):
        h, w = img.shape
        min_w, min_h = min_size
        max_w, max_h = max_size
        while w >= min_w and h >= min_h:
            if w <= max_w and h <= max_h:
                yield img
            w /= scale_factor
            h /= scale_factor
            img = cv.resize(img, (int(w), int(h)),
                             interpolation=cv.INTER_AREA)
    def resizeInput(self, img, limit):
        if (img.shape[0] > limit):
            factor = float(limit) / float(img.shape[0])
            factor = factor * factor
            img = cv.resize(img, (0, 0), fx = factor, fy = factor)
        return img

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
