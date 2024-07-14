import cv2 as cv
import numpy as np
import os    
from non_max_suppression import non_max_suppression_fast as nms

CUM_DATA_PATH = '../../Data/TestPhotos/Cumulus/'
NEG_DATA_PATH = '../../Data/TestPhotos/NEG/'
TEST_PATH = '../../Data/TestPhotos/BackgroundTest/TEST.jpg'
BOW_NUM_TRAINING_SAMPLES_PER_CLASS = 20
SVM_NUM_TRAINING_SAMPLES_PER_CLASS = 200
CLASSES = ['cumulus','altocumulus','cumulonimbus','stratocumulus','stratus','cirrostratus','cirrus','cirrocumulus']
FLANN_INDEX_KDTREE = 1
SVM_SCORE_THRESHOLD = 2.0
NMS_OVERLAP_THRESHOLD = 0.3

inputImage = None

sift = cv.xfeatures2d.SIFT_create()
                
if not os.path.exists(TEST_PATH):
    print("Couldn't find input image file")
else:
    inputImage = cv.imread(TEST_PATH)

    if not os.path.isdir(CUM_DATA_PATH):
        print('data not found')
        exit(1)
    else:
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = {}
        flann = cv.FlannBasedMatcher(index_params, search_params)
        bow_kmeans_trainer = cv.BOWKMeansTrainer(5)
        bow_extractor = cv.BOWImgDescriptorExtractor(sift, flann)

        def get_path_data(i):
            pos_path = CUM_DATA_PATH + "Cumulus" + str(i) + "R.JPG"
            neg_path = NEG_DATA_PATH + "NEG" + str(i) + "R.JPG"
            return pos_path, neg_path

        def add_sample(path):
            img = cv.imread(path, cv.IMREAD_GRAYSCALE)
            keypoints, descriptors = sift.detectAndCompute(img, None)
            if descriptors is not None:
                bow_kmeans_trainer.add(descriptors)
        
        for i in range(BOW_NUM_TRAINING_SAMPLES_PER_CLASS):
            pos_path,neg_path = get_path_data(i+1)
            add_sample(pos_path)
            add_sample(neg_path)

        voc = bow_kmeans_trainer.cluster()
        bow_extractor.setVocabulary(voc)

        def extract_bow_descriptors(img):
            features = sift.detect(img)
            return bow_extractor.compute(img, features)

        training_data = []
        training_labels = []

        for i in range(SVM_NUM_TRAINING_SAMPLES_PER_CLASS):
            pos_path, neg_path = get_path_data(i+1)
            pos_img = cv.imread(pos_path,cv.IMREAD_GRAYSCALE)
            pos_descriptors = extract_bow_descriptors(pos_img)
            if pos_descriptors is not None:
                training_data.extend(pos_descriptors)
                training_labels.append(1)
            neg_img = cv.imread(neg_path,cv.IMREAD_GRAYSCALE)
            neg_descriptors = extract_bow_descriptors(neg_img)
            if neg_descriptors is not None:
                training_data.extend(neg_descriptors)
                training_labels.append(-1)

        svm = cv.ml.SVM_create()
        svm.setType(cv.ml.SVM_C_SVC)
        svm.setC(50)
        svm.train(np.array(training_data), cv.ml.ROW_SAMPLE, np.array(training_labels))

        print("SVM READY")

        def pyramid(img, scale_factor=2, min_size=(300, 200),
                    max_size=(3000, 3000)):
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

        def sliding_window(img, step=25, window_size=(150, 100)):
            img_h, img_w = img.shape
            window_w, window_h = window_size
            for y in range(0, img_w, step):
                for x in range(0, img_h, step):
                    roi = img[y:y+window_h, x:x+window_w]
                    roi_h, roi_w = roi.shape
                    if roi_w == window_w and roi_h == window_h:
                        yield (x, y, roi)

        print("Detecting and classifying clouds in sky...")
        img = cv.imread(TEST_PATH)
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        pos_rects = []
        #cv.namedWindow(TEST_PATH, cv.WINDOW_NORMAL)
        for resized in pyramid(gray_img):
            stopper = 0
            print("resized: ", resized.shape)
            for x, y, roi in sliding_window(resized):
                #if stopper >= 20:
                #    break
                #print("inner", stopper)
                #print("X: " + str(x) + ", Y: " + str(y) + ", " + str(stopper))
                descriptors = extract_bow_descriptors(roi)
                if descriptors is None:
                    continue
                #print("descriptors: ", descriptors)
                prediction = svm.predict(descriptors)
                #print(prediction)
                if prediction[1][0][0] == 1.0:
                    raw_prediction = svm.predict(
                        descriptors, 
                        flags=cv.ml.STAT_MODEL_RAW_OUTPUT)
                    
                    score = -raw_prediction[1][0][0]

                    print("raw: ", raw_prediction)
                    
                    if score > SVM_SCORE_THRESHOLD:
                        h, w = roi.shape
                        scale = gray_img.shape[0] / \
                            float(resized.shape[0])
                        pos_rects.append([int(x * scale),
                                          int(y * scale),
                                          int((x+w) * scale),
                                          int((y+h) * scale),
                                          score])
                stopper += 1
        pos_rects = nms(np.array(pos_rects), NMS_OVERLAP_THRESHOLD)
        print('pos rects complete')
        print(pos_rects)
        for x0, y0, x1, y1, score in pos_rects:
            cv.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)),
                          (0, 255, 255), 10)
            text = '%.2f' % score
            cv.putText(img, text, (int(x0), int(y0) - 20),
                            cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 6)
        #cv.namedWindow(TEST_PATH, cv.WINDOW_NORMAL)
        cv.imshow(TEST_PATH, img)
        cv.waitKey(0)
        cv.destroyAllWindows()

        
        
def isolate_sky(originalImg, fg_proportion=0.4):
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
