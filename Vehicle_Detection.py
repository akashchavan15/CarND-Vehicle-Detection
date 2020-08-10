import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from skimage.feature import hog
from scipy.ndimage.measurements import label
import pickle
from moviepy.editor import VideoFileClip
from src.Advanced_Lane_Lines import AdvanceLaneLines


class Vehicle_Detection:
    def __init__(self,color_space):
        """
        Initialize members
        :param color_space: image color space
        """
        self.color_space = color_space  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        # load the model from disk
        filename = 'model.sav'
        self.svc = pickle.load(open(filename, 'rb'))
        #load the scalar from disk
        filename = 'Scalar.sav'
        self.X_scalar = pickle.load(open(filename, 'rb'))
        self.prev_rects = []
        self.lanes = AdvanceLaneLines()

    @staticmethod
    def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False,
                         feature_vec=True):
        """
        Finds hog features in an image
        :param img: input image
        :param orient: HOG orientations
        :param pix_per_cell: HOG pixels per cell
        :param cell_per_block: HOG cells per block
        :param vis: Boolean to get hog image
        :param feature_vec: Boolean to return data as feature vector
        :return: feature vector
        """
        if vis:
            features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block),
                                      transform_sqrt=False, visualize=vis, feature_vector=feature_vec)

            return features, hog_image
        else:
            features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                           cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                           visualize=vis, feature_vector=feature_vec)

        return features

    @staticmethod
    def bin_spatial(img, size=(32, 32)):
        """
        Compute binned color features
        :param img:input image
        :param size:size of the image
        :return:stacked color features
        """
        color1 = cv2.resize(img[:, :, 0], size).ravel()
        color2 = cv2.resize(img[:, :, 1], size).ravel()
        color3 = cv2.resize(img[:, :, 2], size).ravel()
        return np.hstack((color1, color2, color3))

    @staticmethod
    def color_hist(img, nbins=32):
        """
        Compute color histogram features
        :param img: input color image
        :param nbins: number of histogram bins
        :return: feature vector containing color histograms
        """
        rhist = np.histogram(img[:, :, 0], bins=nbins)
        ghist = np.histogram(img[:, :, 1], bins=nbins)
        bhist = np.histogram(img[:, :, 2], bins=nbins)
        hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
        return hist_features

    def convert_color(self, img):
        """
        Converts input image's color space
        :param img: input image
        :return: color space converted image
        """
        if self.color_space == 'RGB2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        if self.color_space == 'BGR2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        if self.color_space == 'RGB2LUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        if self.color_space == 'BGR2HSV':
            return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if self.color_space == 'RGB2YUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        if self.color_space == 'RGB2HLS':
            return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    def find_cars(self, img, ystart, ystop, scale, orient, pix_per_cell, cell_per_block, spatial_size,
                  hist_bins):
        """
        Function that can extract features using hog sub-sampling and make predictions
        :param img: input image
        :param ystart: window start y position
        :param ystop: window stop y position
        :param scale: scale of window
        :param orient: HOG orientations
        :param pix_per_cell: HOG pixels per cell
        :param cell_per_block: HOG cells per block
        :param spatial_size: size to resize input image
        :param hist_bins: number of histogram bins
        :return: rectangles containing found vehicle locations
        """
        count = 0
        img = img.astype(np.float32) / 255  # Important because we trained on png and now loadin jpeg image
        # PNG are read in 0 to 1 and jpeg in 0 to 255
        img_tosearch = img[ystart:ystop, :, :]
        ctrans_tosearch = self.convert_color(img_tosearch)
        rectangles = []
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))
        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) + 1  # -1 // to get the integer result
        nyblocks = (ch1.shape[0] // pix_per_cell) + 1  # -1
        nfeat_per_block = orient * cell_per_block ** 2

        window = 64
        nblocks_per_window = (window // pix_per_cell) - 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        hog1 = self.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = self.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = self.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                count += 1
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos * pix_per_cell
                ytop = ypos * pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))
                # Get color features
                spatial_features = self.bin_spatial(subimg, size=spatial_size)
                hist_features = self.color_hist(subimg, nbins=hist_bins)
                # Scale features and make a prediction
                test_features = self.X_scalar.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))

                test_prediction = self.svc.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)
                    rectangles.append(
                        ((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))

        return rectangles

    @staticmethod
    def apply_threshold(heatmap, threshold):
        """
        Applies threshold to heatmap
        :param heatmap: input heatmap of detection
        :param threshold: threshold
        :return:
        """
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap

    @staticmethod
    def add_heat(heatmap, bbox_list):
        """
        Updates heatmap
        :param heatmap: input heatmap
        :param bbox_list: list of bounding boxes
        :return: updated heatmap
        """
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        return heatmap

    @staticmethod
    def draw_labeled_bboxes(img, labels):
        """
        Draws label boxes on input image
        :param img: input image
        :param labels: labels
        :return: output image and rectangles
        """
        # Iterate through all detected cars
        rects = []
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            rects.append(bbox)
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
        # Return the image and final rectangles
        return img, rects

    def add_rects(self, rects):
        """
        Adds new rectangles to previous
        :param rects: input rectangles
        :return: None
        """
        self.prev_rects.append(rects)
        if len(self.prev_rects) > 15:
            # throw out oldest rectangle set(s)
            self.prev_rects = self.prev_rects[len(self.prev_rects) - 15:]

    def process_frame(self, img):
        """
        Processes input image
        :param img: input image
        :return: output image with detected vehicles
        """
        rectangles = []
        orient = 9  # HOG orientations
        pix_per_cell = 8  # HOG pixels per cell
        cell_per_block = 2  # HOG cells per block
        spatial_size = (32, 32)  # Spatial binning dimensions
        hist_bins = 32  # Number of histogram bins

        ystart = [400, 416, 400, 432, 400, 432, 400, 464]
        ystop = [464, 480, 496, 528, 528, 560, 596, 660]
        scale = [1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 3.5, 3.5]

        for i in range(len(ystart)):
            rectangles.append(
                self.find_cars(img=img,ystart=ystart[i], ystop=ystop[i], scale=scale[i],
                          orient=orient, pix_per_cell=pix_per_cell, cell_per_block=
                          cell_per_block, spatial_size=spatial_size, hist_bins=hist_bins))

        rectangles = [item for sublist in rectangles for item in sublist]
        # add detections to the history
        if len(rectangles) > 0:
            self.add_rects(rectangles)

        heatmap_img = np.zeros_like(img[:, :, 0])
        for rect_set in self.prev_rects:
            heatmap_img = self.add_heat(heatmap_img, rect_set)
        heatmap_img = self.apply_threshold(heatmap_img, 1 + len(self.prev_rects) // 2)

        labels = label(heatmap_img)
        draw_img, rect = self.draw_labeled_bboxes(np.copy(img), labels)
        return draw_img

    def __repr__(self):
        return "Pipeline('{}')".format(self.color_space)

def process_test_images(vehicle):
    """
    Processes test images
    :param vehicle: Vehicle Object
    :return: None
    """
    search_path = 'test_images\*'
    example_images = glob.glob(search_path)

    # Iterate over the test image
    for img_src in example_images:
        #img = cv2.imread(img_src)
        img = mpimg.imread(img_src)
        lane_img = vehicle.lanes.get_lanes_lines(img)
        od_img = vehicle.process_frame(img)
        result = cv2.addWeighted(od_img, 1, lane_img, 0.6, 0)  # this is commented for project 5
        plt.imshow(result)
        plt.show()

def process_video_images(img,vehicle):
    """
    Processes input images coming from video
    :param img: input image
    :param vehicle: Vehicle object
    :return: output image with overlay
    """
    vehicle_out = vehicle.process_frame(img.copy())
    lanes_out = vehicle.lanes.get_lanes_lines(img.copy())
    result = cv2.addWeighted(vehicle_out, 1, lanes_out, 0.6, 0)

    return result


def main():
    test_images = False
    project_video = True
    vehicle = Vehicle_Detection('RGB2YCrCb')
    print(vehicle)
    if test_images:
        process_test_images(vehicle)

    if project_video:
        white_output = 'output_images/project_video.mp4'
        clip1 = VideoFileClip("project_video.mp4")
        white_clip = clip1.fl_image(lambda image:process_video_images(image,vehicle))  # NOTE: this function expects color images!!
        white_clip.write_videofile(white_output, audio=False)

if __name__ == "__main__":
    main()
