# Based on https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/raspberry_pi/README.md
import re
import cv2
from tflite_runtime.interpreter import Interpreter
import numpy as np
#from joblib import load
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


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
    while cap.isOpened():
        ret, frame = cap.read()
        img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (320,320))
        res = detect_objects(interpreter, img, 0.8)
        print(res)

        for result in res:
            ymin, xmin, ymax, xmax = result['bounding_box']
            xmin = int(max(1,xmin * CAMERA_WIDTH))
            xmax = int(min(CAMERA_WIDTH, xmax * CAMERA_WIDTH))
            ymin = int(max(1, ymin * CAMERA_HEIGHT))
            ymax = int(min(CAMERA_HEIGHT, ymax * CAMERA_HEIGHT))
            
            cv2.rectangle(frame,(xmin, ymin),(xmax, ymax),(0,255,0),3)

            cv2.putText(frame,labels[int(result['class_id'])],(xmin, min(ymax, CAMERA_HEIGHT-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2,cv2.LINE_AA) 

        cv2.imshow('Pi Feed', frame)

        if cv2.waitKey(10) & 0xFF ==ord('q'):
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()