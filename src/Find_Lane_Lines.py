import cv2
import numpy as np

class Find_Lane_Lines:
    def __init__(self):
        self.left_line_x_coord = [] # x coordinates of left lane line
        self.left_line_y_coord = [] # y coordinates of left lane line
        self.right_line_x_coord = [] # x coordinates of right lane line
        self.right_line_y_coord = [] # y coordinates of right lane line
        self.first_image = True  # Flag to run sliding window only once
        self.left_poly_coeffs = [] # Coefficients of left line polynomial
        self.right_poly_coeffs = [] # Coefficients of right line polynomial
        self.points_y = [] # Points to draw overlays
        self.ym_per_pix = 3.048/100 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/278 # meters per pixel in x dimension
        self.debug = False # Flag to get additional output for debugging


    def __find_lane_pixels(self,bin_warped_img,col_warped_img = None):
        """
        Function to find pixels which belongs to left and right lane lines
        :param bin_warped_img:binary input warped image
        :return:
        """
        viz_img = np.dstack((bin_warped_img, bin_warped_img, bin_warped_img))
        if self.debug:
            viz_img = np.copy(col_warped_img)

        # Eliminate not interested region
        bin_warped_img[:, 0:150] = 0
        bin_warped_img[:, 1200:] = 0
        bin_warped_img[0:200, :] = 0

        # Take a histogram of the bottom half of the image
        hist = np.sum(bin_warped_img[:bin_warped_img.shape[0], :], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        mid = np.int(hist.shape[0] // 2)
        left_x_base = np.argmax(hist[:mid])
        right_x_base = np.argmax(hist[mid:]) + mid
        n_windows = 9 # Number of sliding windows
        margin = 150 # Width of the window
        min_px = 25 # Minimum number of pixels to recenter the window
        w_ht = np.int(bin_warped_img.shape[0] // n_windows) # set height of window based upon number of windows
                                                          # and image height
        # Identify the x and y positions of all nonzero pixels in the image
        non_zero = bin_warped_img.nonzero()
        non_zero_y = np.array(non_zero[0])
        non_zero_x = np.array(non_zero[1])

        # Current positions to be updated later for each window in nwindows
        left_x_current = left_x_base
        right_x_current = right_x_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        # Step through the windows one by one
        for window in range(n_windows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = bin_warped_img.shape[0] - (window + 1) * w_ht
                win_y_high = bin_warped_img.shape[0] - window * w_ht
                win_xleft_low = left_x_current - margin
                win_xleft_high = left_x_current + margin
                win_xright_low = right_x_current - margin
                win_xright_high = right_x_current + margin

                if self.debug:
                    # Draw the windows on the visualization image
                    cv2.rectangle(viz_img, (win_xleft_low, win_y_low),
                                  (win_xleft_high, win_y_high), (0, 255, 0), 4)
                    cv2.rectangle(viz_img, (win_xright_low, win_y_low),
                                  (win_xright_high, win_y_high), (0, 255, 0), 4)

                # Identify the nonzero pixels in x and y within the window
                good_left_inds = ((non_zero_y >= win_y_low) & (non_zero_y < win_y_high) &
                                  (non_zero_x >= win_xleft_low) & (non_zero_x < win_xleft_high)).nonzero()[0]
                good_right_inds = ((non_zero_y >= win_y_low) & (non_zero_y < win_y_high) &
                                   (non_zero_x >= win_xright_low) & (non_zero_x < win_xright_high)).nonzero()[0]

                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)

                # If we find pixels more than min_px, recenter next window on its mean position
                if len(good_left_inds) > min_px:
                    left_x_current = np.int(np.mean(non_zero_x[good_left_inds]))
                if len(good_right_inds) > min_px:
                    right_x_current = np.int(np.mean(non_zero_x[good_right_inds]))
        # Concatenate the arrays of indices
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Do nothing
            pass
        # Extract left and right line pixel positions
        self.left_line_x_coord = non_zero_x[left_lane_inds]
        self.left_line_y_coord = non_zero_y[left_lane_inds]
        self.right_line_x_coord = non_zero_x[right_lane_inds]
        self.right_line_y_coord = non_zero_y[right_lane_inds]

        if self.debug:
            return viz_img

    def fit_polynomial(self,Minv,bin_warped_img,undist_img,col_warped_img = None):
        """
        Fits the polynomial on the left and right lane line coordinates
        :param bin_warped_img: binary warped image
        :param undist_img: original undistorted image
        :param Minv: inverse M matrix to dewarp image
        :param col_warped_img: warped color image
        :return: weighted warped color image with overlays
        """
        # find lane pixels first
        if self.debug:
            sliding_window_img = self.__find_lane_pixels(bin_warped_img,col_warped_img)
        else:
            _ = self.__find_lane_pixels(bin_warped_img)

        # fit a second order polynomial
        self.left_poly_coeffs = np.polyfit(self.left_line_y_coord,self.left_line_x_coord,2)
        self.right_poly_coeffs = np.polyfit(self.right_line_y_coord,self.right_line_x_coord,2)

        # Generate points for drawing
        self.points_y = np.linspace(0, bin_warped_img.shape[0] - 1, bin_warped_img.shape[0])
        # find all x points between two lane lines
        try:
            left_fit_x = self.left_poly_coeffs[0] * self.points_y ** 2 + self.left_poly_coeffs[1] * self.points_y + self.left_poly_coeffs[2]
            right_fit_x = self.right_poly_coeffs[0] * self.points_y ** 2 + self.right_poly_coeffs[1] * self.points_y + self.right_poly_coeffs[2]
        except:
            print('Failed to fit a line!')
            left_fit_x = 1 * self.points_y ** 2 + 1 * self.points_y
            right_fit_x = 1 * self.points_y ** 2 + 1 * self.points_y

        # Create an image to draw the lines on
        warp_zero_img = np.zeros_like(bin_warped_img).astype(np.uint8)
        color_warp_img = np.dstack((warp_zero_img, warp_zero_img, warp_zero_img))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fit_x, self.points_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, self.points_y])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp_img, np.int_([pts]), (0, 255, 0))
        cv2.polylines(color_warp_img, np.int32([pts_left]), isClosed=False, color=(255, 0, 0), thickness=15)
        cv2.polylines(color_warp_img, np.int32([pts_right]), isClosed=False, color=(255, 0, 0), thickness=15)

        final_warped_img = cv2.warpPerspective(color_warp_img, Minv, (undist_img.shape[1], undist_img.shape[0]))
        # Combine the result with the original image
        final_warped_img = cv2.addWeighted(undist_img, 1, final_warped_img, 0.6, 0)
        left_curverad, right_curverad, center_dist = self.__calculate_radius_of_curvature(bin_warped_img)
        final_warped_img = self.__draw_data(final_warped_img,left_curverad, right_curverad, center_dist)

        if self.debug:
            return sliding_window_img,final_warped_img

        self.first_image = False
        return final_warped_img

    def search_around_poly(self,Minv,bin_warped_img,undist_img):
        """
        Function to search lane lines pixels around already found lane lines
        :param Minv: inverse M matrix to dewarp image
        :param bin_warped_img:binary warped image
        :param undist_img:original undistorted image
        :return:weighted warped color image with overlays
        """

        margin = 50
        # Grab activated pixels
        non_zero = bin_warped_img.nonzero()
        non_zero_y = np.array(non_zero[0])
        non_zero_x = np.array(non_zero[1])

        #Set the area of search based on activated x-values
        left_lane_inds = ((non_zero_x > (self.left_poly_coeffs[0] * (non_zero_y ** 2) + self.left_poly_coeffs[1] * non_zero_y +
                                         self.left_poly_coeffs[2] - margin)) & (non_zero_x < (self.left_poly_coeffs[0] * (non_zero_y ** 2) +
                                         self.left_poly_coeffs[1] * non_zero_y + self.left_poly_coeffs[2] + margin)))

        right_lane_inds = ((non_zero_x > (self.right_poly_coeffs[0] * (non_zero_y ** 2) + self.right_poly_coeffs[1] * non_zero_y +
                                          self.right_poly_coeffs[2] - margin)) & (non_zero_x < (self.right_poly_coeffs[0] * (non_zero_y ** 2) +
                                          self.right_poly_coeffs[1] * non_zero_y + self.right_poly_coeffs[2] + margin)))

        # Again, extract left and right line pixel positions
        self.left_line_x_coord = non_zero_x[left_lane_inds]
        self.left_line_y_coord = non_zero_y[left_lane_inds]
        self.right_line_x_coord = non_zero_x[right_lane_inds]
        self.right_line_y_coord = non_zero_y[right_lane_inds]

        # Fit new polynomials
        left_fit_x, right_fit_x, points_y = self.fit_poly(bin_warped_img.shape, self.left_line_x_coord, self.left_line_y_coord,
                                                     self.right_line_x_coord, self.right_line_y_coord)

        # Create image to draw overlay on
        viz_img = np.dstack((bin_warped_img, bin_warped_img, bin_warped_img)) * 255
        window_img = np.zeros_like(viz_img)
        # Color in left and right line pixels
        window_img[non_zero_y[left_lane_inds], non_zero_x[left_lane_inds]] = [255, 0, 0]
        window_img[non_zero_y[right_lane_inds], non_zero_x[right_lane_inds]] = [255, 0, 0]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fit_x - margin, points_y]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fit_x + margin,
                                                                        points_y])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))

        right_line_window1 = np.array([np.transpose(np.vstack([right_fit_x - margin, points_y]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fit_x + margin,
                                                                         points_y])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        if self.debug:
            cv2.fillPoly(viz_img, np.int_([left_line_pts]), (0, 255, 0))
            cv2.fillPoly(viz_img, np.int_([right_line_pts]), (0, 255, 0))
            result = cv2.addWeighted(viz_img, 1, window_img, 0.3, 0)

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(bin_warped_img).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fit_x, points_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, points_y])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255, 0, 0), thickness=15)
        cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(255, 0, 0), thickness=15)
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (undist_img.shape[1], undist_img.shape[0]))
        # Combine the result with the original image
        final_img = cv2.addWeighted(undist_img, 1, newwarp, 0.6, 0)

        left_curverad, right_curverad, center_dist = self.__calculate_radius_of_curvature(bin_warped_img)
        final_img = self.__draw_data(final_img,left_curverad, right_curverad, center_dist)

        if self.debug:
            return result,final_img

        return final_img

    def fit_poly(self,img_shape,left_x,left_y,right_x,right_y):
        """
        Fits the polynomial on the x and y coordinates of the lane lines
        :param img_shape:shape of input image
        :param left_x:x coordinates of left line points
        :param left_y:y coordinates of left line points
        :param right_x:x coordinate of right line points
        :param right_y:y coordinates of right line points
        :return:points between left and right lane lines
        """
        ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
        temp1 = np.isinf(left_y).any()
        temp2 = np.isinf(left_x).any()
        temp3 = np.isnan(left_x).any()
        temp4 = np.isnan(left_y).any()
        self.left_poly_coeffs = np.polyfit(left_y , left_x , 2)
        self.right_poly_coeffs = np.polyfit(right_y , right_x , 2)
        # Generate x and y values for plotting
        #ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
        ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
        left_fitx = self.left_poly_coeffs[0] * self.points_y ** 2 + self.left_poly_coeffs [1] * self.points_y + self.left_poly_coeffs [2]
        right_fitx = self.right_poly_coeffs[0] * self.points_y  ** 2 + self.right_poly_coeffs[1] * self.points_y  + self.right_poly_coeffs[2]

        return left_fitx, right_fitx, self.points_y

    def __calculate_radius_of_curvature(self,bin_warped_img):
        """
        Calculates the radius of curvature and center distance
        :param bin_warped_img:input binary warped image
        :return:left, right curvature radii and center distance
        """
        ht = bin_warped_img.shape[0]
        # choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(self.points_y)

        if len(self.left_line_x_coord) != 0 and len(self.right_line_x_coord) != 0:
            # Fit new polynomials to x,y in world space
            left_fit_cr = np.polyfit(self.left_line_y_coord * self.ym_per_pix, self.left_line_x_coord * self.xm_per_pix, 2)
            right_fit_cr = np.polyfit(self.right_line_y_coord * self.ym_per_pix, self.right_line_x_coord * self.xm_per_pix, 2)

        # Calculation of R_curve (radius of curvature)
        left_curve_rad = ((1 + (2 * left_fit_cr[0] * y_eval * self.ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curve_rad = ((1 + (2 * right_fit_cr[0] * y_eval * self.ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])

        # Distance from center is image x midpoint - mean of l_fit and r_fit intercepts
        if self.right_poly_coeffs is not None and self.left_poly_coeffs  is not None:
            car_position = bin_warped_img.shape[1]/2
            l_fit_x_int = self.left_poly_coeffs [0]*ht**2 + self.left_poly_coeffs [1]*ht + self.left_poly_coeffs [2]
            r_fit_x_int = self.right_poly_coeffs[0]*ht**2 + self.right_poly_coeffs[1]*ht + self.right_poly_coeffs[2]
            lane_center_position = (r_fit_x_int + l_fit_x_int) /2
            center_dist = (car_position - lane_center_position) * self.xm_per_pix

        return left_curve_rad, right_curve_rad, center_dist

    def __draw_data(self,final_img, left_curve_rad,right_curve_rad , center_dist):
        """
        draws text on final image
        :param final_img: input final image
        :param left_curve_rad: left line radius of curvature
        :param right_curve_rad: right line radius of curvature
        :param center_dist: center distance
        :return:
        """
        new_img = np.copy(final_img)
        font = cv2.FONT_HERSHEY_DUPLEX
        text_left = 'Left Curvature: ' + '{:04.2f}'.format(left_curve_rad) + 'm'
        text_right = 'Right Curvature: ' + '{:04.2f}'.format(right_curve_rad) + 'm'
        cv2.putText(new_img, text_left, (40,70), font, 1.5, (200,255,155), 2, cv2.LINE_AA)
        cv2.putText(new_img, text_right, (40, 120), font, 1.5, (200, 255, 155), 2, cv2.LINE_AA)
        direction = ''
        if center_dist > 0:
            direction = 'right'
        elif center_dist < 0:
            direction = 'left'
        abs_center_dist = abs(center_dist)
        text = '{:04.3f}'.format(abs_center_dist) + 'm ' + direction + ' of center'
        cv2.putText(new_img, text, (40,190), font, 1.5, (200,255,155), 2, cv2.LINE_AA)

        return new_img


