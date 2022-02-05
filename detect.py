# Based on https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/raspberry_pi/README.md
import re
import cv2
from tflite_runtime.interpreter import Interpreter
import numpy as np
#from joblib import load
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


kernel_sharpen = np.array([0,-1,-0],[-1,5,-1],[0,-1,0])
# Shparen use filter2D(img,-1,kểnl)
kernel_unsharp_masking = 1 / (256*np.array([1,4,6,4,1],[4,16,24,16,4],[6,24,-476,24,6],[4,16,24,16,4],[1,4,6,4,1]))
# Unsharp_masking based on gaussian blur and amount =1 and threshold =0

def load_labels(path='labels.txt'):
  """Loads the labels file. Supports files with or without index numbers."""
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    labels = {}
    for row_number, content in enumerate(lines):
      pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
      if len(pair) == 2 and pair[0].strip().isdigit():
        labels[int(pair[0])] = pair[1].strip()
      else:
        labels[row_number] = pair[0].strip()
  return labels

def set_input_tensor(interpreter, image):
  """Sets the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = np.expand_dims((image-255)/255, axis=0)


def get_output_tensor(interpreter, index):
  """Returns the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  # Get all output details
  boxes = get_output_tensor(interpreter, 1)
  classes = get_output_tensor(interpreter, 3)
  scores = get_output_tensor(interpreter, 0)
  count = int(get_output_tensor(interpreter, 2))

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
          'bounding_box': boxes[i],
          'class_id': classes[i],
          'score': scores[i]
      }
      results.append(result)
  return results

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


def main():
    labels = load_labels()
    interpreter = Interpreter('detect.tflite')
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    cap = cv2.VideoCapture(0)
    wi = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    he = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while cap.isOpened():
        ret, frame = cap.read()
        img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (320,320))
        res = detect_objects(interpreter, img, 0.8)
        print(res)

        for result in res:
            # Tọa độ tương đối
            ymin, xmin, ymax, xmax = result['bounding_box']
            # Tọa độ tuyệt đối theo frame và chuẩn hóa để ko lỗi ngoài frame
            xmin = int(max(1,xmin * wi))
            xmax = int(min(wi, xmax * wi))
            ymin = int(max(1, ymin * he))
            ymax = int(min(he, ymax * he))

            cv2.rectangle(frame,(xmin, ymin),(xmax, ymax),(0,255,0),3)
            img_ob = frame[ymin:ymax,xmin:xmax]
            img_gray = cv2.cvtColor(img_ob,cv2.COLOR_BGR2GRAY)
            ob1 = img_gray.copy()
            cv2.imshow('ob',ob1)
            #ob1 = cv2.resize(ob1,(w_resize,h_resize))
            #gray = cv2.medianBlur(ob1,5)
            #gray = cv2.bilateralFilter(ob1,9,75,75)
            #gray = cv2.cvtColor(ob1,cv2.COLOR_BGR2GRAY)
            edge_detec = cv2.Canny(ob1, 30, 200,None,None,True)
            kernel2= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))
            kernel3= cv2.getStructuringElement(cv2.MORPH_RECT,(4,4))
            edge_detec = cv2.erode(edge_detec,kernel2,iterations=1)
            edge_detec = cv2.dilate(edge_detec,kernel3,iterations=1)
            cv2.imshow('edge', edge_detec)
            contours, hierarchy = cv2.findContours(edge_detec, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Chọn Ob to nhất
            tff = 0
            area_save = 0
            for i in range(len(contours)):
                if cv2.contourArea(contours[i]) > area_save:
                    area_save = cv2.contourArea(contours[i])
                    tff = i
            # print('tf:', tff)
            cv2.drawContours(ob1, contours,tff,(255,0,0),2)
            
            # Xoay ngang
            rect = cv2.minAreaRect(contours[tff])
            img_croped = crop_minAreaRect(img_ob,rect)
            cv2.imshow('croped',img_croped)

            # Lấy Feature mình cần
            h,w = img_croped.shape[:2]
            CP = img_croped[0:h,0:round(0.1*w)]
            if CP is not None:
                
                CO = CP.copy()
                ret,thresh1 = cv2.threshold(CO,130,255,cv2.THRESH_BINARY_INV)
                kernel2= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))     #9,10 cho KQ OKE
                kernel3= cv2.getStructuringElement(cv2.MORPH_RECT,(11,11)) # khả thi dải ksize rộng
                erosion = cv2.erode(thresh1,kernel2,iterations = 1)
                dilation = cv2.dilate(erosion,kernel3,iterations = 1)
                contours,hierarchy= cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for i in range(len(contours)):
                    cv2.drawContours(CP, contours,i,(255,0,0),3)
                    print("Cout: ", str(len(contours) ))
                    cv2.imshow('OriIMG',CP)
            '''
            if len(contours) != (6 or 4):
                print('NG')
            else: print('OK')
            cv2.putText(frame,labels[int(result['class_id'])],(xmin, min(ymax, CAMERA_HEIGHT-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2,cv2.LINE_AA) 
            '''
        cv2.imshow('Pi Feed', frame)

        if cv2.waitKey(10) & 0xFF ==ord('q'):
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()