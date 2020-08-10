import numpy as np
import cv2

class Transform_Image:
    def __init__(self):
        self.src = []  # source points for transform
        self.dst = [] # destination points for transform
        self.M = 0 # Perspective Transform Matrix
        self.invM = 0 # Inverse Perspective Transform Matrix
        self.points_loaded = False

    def undistort(self, img, mtx, dist):
        """
        Corrects image distortion using calibration matrices
        :param img:
        :return:
        """
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        return undist

    def warp_image(self,bin_thresholded_img,undist_img):
        """
        Warps the input image
        :param threshold_img:Gradient and Color Thresholded image
        :return:warped image
        """
        if not self.points_loaded:
        # define source and destination points for transform
            bottomY = 672 # originally it was set to 720
            topY = 440

            bottom_left = (190, bottomY) # bottom_left
            top_left = (585, topY)
            top_right = (705, topY)
            bottom_right = (1130, bottomY)

            # top left -> top right -> bottom left -> bottom right
            src = np.float32([
                top_left,
                top_right,
                bottom_right,
                bottom_left
            ])
            nX = undist_img.shape[1]
            nY = undist_img.shape[0]
            img_size = (nX, nY)
            offset = 200
            dst = np.float32([
                [offset, 0],
                [img_size[0] - offset, 0],
                [img_size[0] - offset, img_size[1]],
                [offset, img_size[1]]
            ])
            self.src = src
            self.dst = dst
            self.points_loaded = True

        img_size = (bin_thresholded_img.shape[1], bin_thresholded_img.shape[0])
        # Given src and dst points, calculate the perspective transform matrix
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        # Warp the image using OpenCV warpPerspective()
        bin_warped = cv2.warpPerspective(bin_thresholded_img, self.M, img_size)

        color_warped = cv2.warpPerspective(undist_img, self.M, img_size)
        self.invM = cv2.getPerspectiveTransform(self.dst, self.src)

        return bin_warped,color_warped