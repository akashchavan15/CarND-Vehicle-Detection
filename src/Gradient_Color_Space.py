import cv2
import numpy as np

class Gradient_Color_Space:

    def __mag_dir_thresh(self,img, sobel_kernel=3):
        """
        Computes the overall magnitude of the gradient
        :param img: input image
        :param sobel_kernel: kernel size
        :return: gradient image
        """
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255
        gradmag = (gradmag/scale_factor).astype(np.uint8)
        # Take the absolute value of the gradient direction
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = self.__apply_threshold(gradmag,absgraddir)

        # Return the binary image
        return binary_output

    def __apply_threshold(self,gradient_img,grad_dir):
        """
        Function to find the gradient thresholds automatically
        :param gradient_img:
        :return: Gradient thresholded binary image
        """
        hist = cv2.calcHist([gradient_img],[0],None,[16],[0,256])
        lower_med = np.median(hist[:7])
        lower_med_idx = np.where(hist == lower_med)
        lower_thershold = lower_med_idx[0][0] * 15  # Multiplying by 15 since we we have 16 total bins

        upper_med = np.median(hist[7:])
        upper_med_idx = np.where(hist == upper_med)
        upper_thershold = upper_med_idx[0][0] * 15 # Multiplying by 15 since we we have 16 total bins

        binary_output = np.zeros_like(gradient_img)
        binary_output[(gradient_img >= lower_thershold) & (gradient_img <= upper_thershold) &
                      (grad_dir <= 1.3) & (grad_dir >= 0.3)] = 1
        return binary_output

    def __hls_select(self,img, thresh=(0, 255)):
        """
        Function to get binary image of S channel of input image
        :param img: input RGB image
        :param thresh: Thresholds for S channel
        :return: binary image of S channel
        """
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]
        binary_output = np.zeros_like(s_channel)
        binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
        return binary_output

    def grad_clr_threshold(self,img):
        """
        Function to find gradient and color threshold of an input image
        :param img: input RGB image
        :return: color binary image and combined gradient and color thresholded image
        """
        s_binary = self.__hls_select(img.copy(),thresh=(100,255))
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        l_channel = hls[:,:,1]
        grad_binary = self.__mag_dir_thresh(l_channel.copy(),5)
        # Stack each channel
        # The green is the gradient threshold component and blue is the color channel threshold component
        color_binary = np.dstack((np.zeros_like(grad_binary), grad_binary, s_binary)) * 255
        # Combine the two binary thresholds
        combined_binary = np.zeros_like(grad_binary)
        combined_binary[(s_binary == 1) | (grad_binary == 1)] = 1
        return color_binary,combined_binary

    def grad_clr_threshold_old(self,img_undistorted, s_thresh=(100, 255), sx_thresh=(30, 100)):
        """
        Uses color transforms, gradients, etc., to create a thresholded binary image.
        :param img: undistorted image
               s_thresh = S color channel threshold (Blue)
               sx_thresh = Gradient threshold in x direction
        :return: thresholded binary image
        """
        # The green is the gradient threshold component and blue is the color channel threshold component
        img = np.copy(img_undistorted)
        # Convert to HLS color space and separate the V channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
        abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        # Stack each channel
        # The green is the gradient threshold component and blue is the color channel threshold component
        color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

        # Combine the two binary thresholds
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

        return color_binary,combined_binary
