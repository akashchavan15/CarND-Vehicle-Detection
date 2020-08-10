from src import Camera_Calibration as cc
from src import Transform_Image as ti
from src import Gradient_Color_Space as gcp
from src import Find_Lane_Lines as fll
import cv2

class AdvanceLaneLines:
    def __init__(self):
        """
        Instantiate required objects
        """
        self.cc = cc.Camera_Calibration()
        self.ti = ti.Transform_Image()
        self.gcs = gcp.Gradient_Color_Space()
        self.fll = fll.Find_Lane_Lines()
        self.calibration_finished = False

    def __covert_img(self,img):
        """
        convert input BGR image to RGB
        :param img: input BGR image
        :return: RGB image
        """
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __calibrate_camera(self,img):
        """
        Function to calibrate camera
        :param img: known input image
        :return:
        """
        self.cc.calibrate_camera(img,8,6,load_saved_points=True)

    def __correct_distortion(self,img):
        """
        Corrects the distortion of an input image using camera calibration matrices
        :param img: distorted input image
        :return: undistorted image
        """
        undist_img = self.ti.undistort(img,self.cc.mtx,self.cc.dist)
        return undist_img


    def __transform_image(self, bin_thresholded_img,undist_img):
        """
        Transforms image by doing Perspective Transformation
        :param img: undistorted input image
        :return: warped image
        """
        bin_warped,color_warped = self.ti.warp_image(bin_thresholded_img,undist_img)
        return bin_warped,color_warped

    def __color_gradient_threshold(self,img):
        """
        Function to perform thresholding on input image based upon gradient
        and color space
        :param img: input image
        :return: thresholded images
        """
        clr_binary, cmb_binary = self.gcs.grad_clr_threshold(img)
        return cmb_binary

    def __find_lane_lines(self,bin_warped_img,undist_img,col_warped_img = None):
        """
        Finds lane lines in input image
        :param bin_warped_img: binary warped image
        :param undist_img: undistorted image
        :param col_warped_img: color warped image
        :return: output image with lane lines 
        """
        if self.fll.debug:
           if self.fll.first_image:
               sliding_window_img,final_warped_img = self.fll.fit_polynomial(self.ti.invM,bin_warped_img,undist_img,col_warped_img)
               return sliding_window_img,final_warped_img
           else:
               col_warped_img,final_warped_img = self.fll.search_around_poly(self.ti.invM,bin_warped_img,undist_img)
               return col_warped_img,final_warped_img
        else:
            if self.fll.first_image:
                final_warped_img = self.fll.fit_polynomial(self.ti.invM,bin_warped_img,undist_img,col_warped_img)
                return final_warped_img
            else:
                final_warped_img = self.fll.search_around_poly(self.ti.invM,bin_warped_img,undist_img)
                return final_warped_img


    def get_lanes_lines(self,img):
        """
        Helper function to process video images through pipeline
        :param img: input image
        :param adv_ll: AdvanceLaneLines object
        :return: output image with lane lines overlays
        """

        img = self.__covert_img(img)
        if not self.calibration_finished:
            self.__calibrate_camera(img)
            self.calibration_finished = True

        undist_img = self.__correct_distortion(img)
        bin_thresholded_img = self.__color_gradient_threshold(undist_img)
        bin_warped_img,color_warped_img = self.__transform_image(bin_thresholded_img,undist_img)
        final_img = self.__find_lane_lines(bin_warped_img,undist_img,color_warped_img)
        final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
        return final_img






