# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run the object detection routine."""
import argparse
import sys
import time
import numpy as np
import cv2
from object_detector import ObjectDetector
from object_detector import ObjectDetectorOptions
import utils

kernel_sharpen = np.array([[0,-1,-0],[-1,5,-1],[0,-1,0]])

def nothing(x):
  pass

def crop_minAreaRect(img, rect):

    # rotate img
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], angle) 
    box = cv2.boxPoints(rect0)
    '''
    for i in range(len(box)):
        for j in range(2):
            if box[i][j] < 0: box[i][j] = 0
    '''
    pts = np.int0(cv2.transform(np.array([box]), M))[0]    
    pts[pts < 0] = 0
    
    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1], 
                       pts[1][0]:pts[2][0]]
    return img_crop

name_window = 'Trackbars'
cv2.namedWindow(name_window)
T1Canny = 30
T2Canny = 200
T1Thres = 125
MaxThres = 255
cv2.createTrackbar('T1Canny', name_window, T1Canny, MaxThres, nothing)
cv2.createTrackbar('T2Canny', name_window, T2Canny, MaxThres, nothing)
cv2.createTrackbar('T1Thres', name_window, T1Thres, MaxThres, nothing)


def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
    model: Name of the TFLite object detection model.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    enable_edgetpu: True/False whether the model is a EdgeTPU model.
  """

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  # Initialize the object detection model
  options = ObjectDetectorOptions(
      num_threads=num_threads,
      score_threshold=0.3,
      max_results=3,
      enable_edgetpu=enable_edgetpu)
  detector = ObjectDetector(model_path=model, options=options)

  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

    counter += 1
    image = cv2.flip(image, 1)

    # Run object detection estimation using the model.
    detections = detector.detect(image)

    # Get Trackbar value
    T1Canny = cv2.getTrackbarPos('T1Canny', name_window)
    T2Canny = cv2.getTrackbarPos('T2Canny', name_window)
    T1Threshold = cv2.getTrackbarPos('T1Threshold', name_window)
    for result in detections:
      ymin, xmin, ymax, xmax = result['bounding_box']
      xmin = int(max(1,xmin * width))
      xmax = int(min(width, xmax * width))
      ymin = int(max(1, ymin * height))
      ymax = int(min(height, ymax * height))

      img_ob = image[ymin:ymax,xmin:xmax]
      img_gray = cv2.cvtColor(img_ob,cv2.COLOR_BGR2GRAY)
      ob1 = img_gray.copy()
      ob1 = cv2.filter2D(ob1,-1,kernel_sharpen)
      edge_detec = cv2.Canny(ob1, T1Canny, T2Canny,None,None,True)
      kernel2= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))
      kernel3= cv2.getStructuringElement(cv2.MORPH_RECT,(4,4))
      edge_detec = cv2.erode(edge_detec,kernel2,iterations=1)
      edge_detec = cv2.dilate(edge_detec,kernel3,iterations=1)
      cv2.imshow('edge', edge_detec)
      contours, hierarchy = cv2.findContours(edge_detec, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

      # Maxx contour
      tff = 0
      area_save = 0
      for i in range(len(contours)):
          if cv2.contourArea(contours[i]) > area_save:
              area_save = cv2.contourArea(contours[i])
              tff = i
      
      #
      rect = cv2.minAreaRect(contours[tff])
      img_croped = crop_minAreaRect(img_ob,rect)
      cv2.imshow('croped',img_croped)

      # Lấy Feature mình cần
      h,w = img_croped.shape[:2]
      CP = img_croped[0:h,0:round(0.1*w)]

      if CP is not None:
                
        CO = CP.copy()
        ret,thresh1 = cv2.threshold(CO,T1Threshold,255,cv2.THRESH_BINARY_INV)
        kernel2= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))     #9,10 cho KQ OKE
        kernel3= cv2.getStructuringElement(cv2.MORPH_RECT,(11,11)) # khả thi dải ksize rộng
        erosion = cv2.erode(thresh1,kernel2,iterations = 1)
        dilation = cv2.dilate(erosion,kernel3,iterations = 1)
        contours,hierarchy= cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            cv2.drawContours(CP, contours,i,(255,0,0),3)
            print("Cout: ", str(len(contours) ))
            cv2.imshow('OriIMG',CP)

    # Draw keypoints and edges on input image
    image = utils.visualize(image, detections)

    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(fps)
    text_location = (left_margin, row_size)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    cv2.imshow('object_detector', image)

  cap.release()
  cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='efficientdet_lite0.tflite')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
  args = parser.parse_args()

  run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads), bool(args.enableEdgeTPU))


if __name__ == '__main__':
  main()
