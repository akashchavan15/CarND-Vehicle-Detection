# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, our goal is to write a software pipeline to detect vehicles in a video provided by Udacity. 

## Pipeline
* Train a SVM classfier by extracting spatial features, color features and Histogram of gradients from the labeled training data set.
* Implement a sliding-window technique and use a trained classifier to predict vehicles in small patches of an image.
* Run the pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map 
  of recurring detections frame by frame to reject outliers and track detected vehicles.
* Draw the bounding on the detected vehicles
* Additionally I have integrated the Advance lane line detection pipeline to detect lane lines in the video. Advance Lane lines detection project can be found
  [here](https://github.com/akashchavan15/CarND-Advanced-Lane-Lines)
