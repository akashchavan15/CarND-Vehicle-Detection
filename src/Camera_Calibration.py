import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import pickle

class Camera_Calibration:
    def __init__(self):
        self.calib_images_loaded = True
        # Arrays to store object points and image points from all the images.
        self.obj_points = []  # 3d points in real world space
        self.img_points = []  # 2d points in image plane.
        self.ret = 0  # RMS re projection error. Should be between 0.1 and 1 pixel. An RMS error of 1.0 means that, on average, each of these projected points is 1.0 px away from its actual position. The error is not bounded in [0, 1], it can be considered as a distance.
        self.mtx = 0  # Camera calibration matrix
        self.dist = 0  # Distortion coefficient
        self.rvecs = 0  # Rotation Vector
        self.tvecs = 0  # Translation Vector
        self.nx = 9
        self.ny = 6

    def __find_points(self):
        """
        Function to get object points and image points using the chessboard images
        :return:
        """
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.ny * self.nx, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1, 2)  # X, Y coordinates

        # Make a list of calibration images
        images = glob.glob('camera_cal/calibration*.jpg')
        fig, axs = plt.subplots(5,4, figsize=(16, 11))
        fig.subplots_adjust(hspace = .2, wspace=.001)
        axs = axs.ravel()

        # Step through the list and search for chessboard corners
        for i, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)

            # If found, add object points, image points
            if ret == True:
                self.obj_points .append(objp)
                self.img_points .append(corners)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (self.nx,self.ny), corners, ret)
                axs[i].axis('off')
                axs[i].imshow(img)
        plt.show()
        self.__save_points()

    def __find_point(self, img,nx, ny):
        """
        Finds image and object points in single image
        :param img: input image
        :param nx: number of inner corners in each row
        :param ny: number of inner corners in each column
        :return:
        """
        self.nx = nx
        self.ny = ny
        self.obj_points.clear()
        self.img_points.clear()
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.ny * self.nx, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1, 2)  # X, Y coordinates
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)
        # If found, add object points, image points
        if ret == True:
            self.obj_points .append(objp)
            self.img_points .append(corners)
            # Draw and display the corners
            #img = cv2.drawChessboardCorners(img, (self.nx,self.ny), corners, ret)

    def __save_points(self):
        """
        This function saves the image points and object points to use them later
        instead of computing every time
        :return:
        """
        file_Name = "imgpoints"
        # open the file for writing
        fileObject = open(file_Name, 'wb')
        pickle.dump(self.img_points, fileObject)
        # close the fileObject
        fileObject.close()
        file_Name = "objpoints"
        # open the file for writing
        fileObject = open(file_Name, 'wb')
        # this writes the object a to the
        pickle.dump(self.obj_points, fileObject)
        # close the fileObject
        fileObject.close()

    def calibrate_camera(self,img, nx, ny, load_saved_points = True):
        """
        Performs the camera calibration
        :param img: input image
        :return:
        """
        img_size = (img.shape[1], img.shape[0])
        if load_saved_points:
            if self.calib_images_loaded:
                # load already saved object points and image points
                self.obj_points = pickle.load( open( "src/objpoints", "rb" ) )
                self.img_points = pickle.load( open( "src/imgpoints", "rb" ) )
                # Do camera calibration given object points and image points
                self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.obj_points,
                                                                        self.img_points, img_size,None,None)
            else:
                self.__find_points()
                # Do camera calibration given object points and image points
                self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.obj_points,
                                                                        self.img_points, img_size,None,None)

        else:
            self.__find_point(img,nx,ny)
            # Do camera calibration given object points and image points
            self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.obj_points,
                                                                    self.img_points, img_size,None,None)
