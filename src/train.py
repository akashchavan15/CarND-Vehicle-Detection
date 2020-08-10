import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import pickle

def load_data():
    """
    Loads a training data
    :return:
    """
    cars = glob.glob('Cars\*.png')
    print('Number of car images',len(cars))
    notcars = glob.glob('NotCars\*.png')
    print('Number of not car images',len(notcars))
    return cars,notcars

def split_dataset(scaled_X, y):
    """
    Splits data set in train and test set
    :param scaled_X: scaled feature vector
    :param y: labels
    :return: train and test data sets
    """
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)
    return X_train, X_test, y_train, y_test

def convert_color(img, conv='RGB2YCrCb'):
    """
    Convert image from one color space to other
    :param img: input space
    :param conv: Output color space
    :return: color space converted image
    """
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'BGR2HSV':
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if conv == 'RGB2YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    if conv == 'RGB2HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

class Object_Detection():
    def __init__(self):
        """
        Initialize members
        """
        self.color_space = 'RGB2YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = 9  # HOG orientations
        self.pix_per_cell = 8 # HOG pixels per cell
        self.cell_per_block = 2 # HOG cells per block
        self.hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
        self.spatial_size = (32, 32) # Spatial binning dimensions
        self.hist_bins = 32    # Number of histogram bins
        self.spatial_feat = True # Spatial features on or off
        self.hist_feat = True # Histogram features on or off
        self.hog_feat = True # HOG features on or off
        self.n_samples = 1000

    def get_hog_features(self,img,vis=False,feature_vec=True):
        """
        Finds hog in an input image
        :param img: input image
        :param vis: Boolean to get hog image
        :param feature_vec: Boolean to return data as feature vector
        :return: feature vector
        """
        if vis:
            features,hog_image = hog(img, orientations=self.orient, pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                                     cells_per_block=(self.cell_per_block, self.cell_per_block),
                                     transform_sqrt=False,
                                      visualize=vis, feature_vector=feature_vec)
            return features, hog_image
        else:
            features = hog(img, orientations=self.orient, pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                           cells_per_block=(self.cell_per_block, self.cell_per_block),
                           transform_sqrt=False,
                           visualize=vis, feature_vector=feature_vec)
            return features

    def bin_spatial(self,img):
        """
        Compute binned color features
        :param img:input image
        :param size:size of the image
        :return:stacked color features
        """
        color1 = cv2.resize(img[:,:, 0],self.spatial_size).ravel()
        color2 = cv2.resize(img[:, :, 1], self.spatial_size).ravel()
        color3 = cv2.resize(img[:, :, 2], self.spatial_size).ravel()
        return np.hstack((color1, color2, color3))

    def color_hist(self,img):
        """
        Compute color histogram features
        :param img: input color image
        :return: feature vector containing color histograms
        """
        # Compute the histogram of the RGB channels separately
        rhist = np.histogram(img[:, :, 0], bins=self.hist_bins)
        ghist = np.histogram(img[:, :, 1], bins=self.hist_bins)
        bhist = np.histogram(img[:, :, 2], bins=self.hist_bins)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
        return hist_features

    def extract_features(self,imgs):
        """
        Extracts features from all input images
        :param imgs: list of images
        :return: list of features
        """
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for file in imgs:
            # Read in each one by one
            file_features = [ ]
            image = mpimg.imread(file)
            # apply color conversion if other than 'RGB'
            if self.color_space != 'RGB':
                feature_image = convert_color(image,self.color_space)
            else:
                feature_image = np.copy(image)

            if self.spatial_feat:
                spatial_features = self.bin_spatial(feature_image)
                file_features.append(spatial_features)

            if self.hist_feat:
                hist_features = self.color_hist(feature_image)
                file_features.append(hist_features)

            if self.hog_feat:
                if self.hog_channel == 'ALL':
                    hog_features = []
                    for channel in range(feature_image.shape[2]):
                        hog_features.append(self.get_hog_features(feature_image[:,:,channel],
                                                                  vis=False,feature_vec= True))
                    hog_features = np.ravel(hog_features)
                else:
                    hog_features = self.get_hog_features(feature_image[:,:,self.hog_channel]
                                                         ,vis=False,feature_vec= True)

                file_features.append(hog_features)
            features.append(np.concatenate(file_features))

        return features

def train_model():
    """
    Trains the model and saves it
    :return:
    """
    cars, non_cars = load_data()
    obj_d = Object_Detection()
    t = time.time()
    cars_features = obj_d.extract_features(cars)
    non_cars_features = obj_d.extract_features(non_cars)
    print(time.time() - t,'seconds to compute features...')
    X = np.vstack((cars_features, non_cars_features)).astype(np.float64)
    # Fit a per column scalar
    X_scalar = StandardScaler().fit(X)
    # save the model to disk
    filename = 'Scalar_v_22.sav'
    pickle.dump(X_scalar, open(filename, 'wb'))
    # Apply the scalar to X
    scaled_X = X_scalar.transform(X)
    # Define the labels vector
    y = np.hstack((np.ones(len(cars_features)), np.zeros(len(non_cars_features))))
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:',obj_d.orient,'orientations',obj_d.pix_per_cell,
          'pixels per cell and', obj_d.cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # save the model to disk
    filename = 'model_v_22.sav'
    pickle.dump(svc, open(filename, 'wb'))

if __name__ == "__main__":
    train_model()
