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
  
### Data Exploration
I began by loading all of the vehicle and non-vehicle image paths from the provided dataset. The figure below shows a random sample of image 
from both classes of the dataset. <br />
<img src="output_images/Car1.png" width="480" alt="Car" /> <img src="output_images/Not_Car1.png" width="480" alt="Non Car" />

### Feature Extraction
#### Color Histogram
RGB image is first converted into YCrCb color space and the color histogram is computed for each color channel for both Car image and Non Car image.  <br />
<img src="output_images/Car1.png" width="480" alt="Car" /> <img src="output_images/Car_Color_Hist1.png" width="480" alt="Non Car" /> <br />
<img src="output_images/Not_Car1.png" width="480" alt="Non Car" /> <img src="output_images/Not_Car_Color_Hist1.png" width="480" alt="Not_Car_Color_Hist" />
