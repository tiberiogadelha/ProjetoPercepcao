import cv2
import numpy as np
import os
import sys
import glob
import tensorflow as tf
import cv2
from matplotlib import pyplot as plt
from distutils.version import StrictVersion
from collections import defaultdict
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


PATH_TO_FROZEN_GRAPH = 'inference_graph/frozen_inference_graph.pb'
PATH_TO_LABELS = 'data/labelmap.pbtxt'
NUM_CLASSES = 37

# Load a (frozen) Tensorflow model into memory.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


def Captcha_detection(image, average_distance_error=3):
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_np = cv2.imread(image)
            image_np = cv2.resize(image_np, (0,0), fx=3, fy=3) 
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            image_np_expanded = np.expand_dims(image_np, axis=0)
     
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        
            (boxes, scores, classes, num_detections) = sess.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
            vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=2)
            cv2.imwrite("captcha_result.png", cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))


            captcha_array = []
            
            for i,b in enumerate(boxes[0]):
                for Symbol in range(37):
                    if classes[0][i] == Symbol: 
                        if scores[0][i] >= 0.55:  
                            mid_x = (boxes[0][i][1]+boxes[0][i][3])/2 # find x coordinates center of letter
                            captcha_array.append([category_index[Symbol].get('name'), mid_x, scores[0][i]]
     
            for number in range(20):
                for captcha_number in range(len(captcha_array)-1):
                    if captcha_array[captcha_number][1] > captcha_array[captcha_number+1][1]:
                        temporary_captcha = captcha_array[captcha_number]
                        captcha_array[captcha_number] = captcha_array[captcha_number+1]
                        captcha_array[captcha_number+1] = temporary_captcha

            average = 0
            captcha_len = len(captcha_array)-1
            while captcha_len > 0:
                average += captcha_array[captcha_len][1]- captcha_array[captcha_len-1][1]
                captcha_len -= 1
            average = average/(len(captcha_array)+average_distance_error)

            
            captcha_array_filtered = list(captcha_array)
            captcha_len = len(captcha_array)-1
            while captcha_len > 0:
                if captcha_array[captcha_len][1]- captcha_array[captcha_len-1][1] < average:
                    if captcha_array[captcha_len][2] >= captcha_array[captcha_len-1][2]:
                        del captcha_array_filtered[captcha_len-1]
                    else:
                        del captcha_array_filtered[captcha_len]
                captcha_len -= 1
                                                 
            captcha_string = ""
            for captcha_letter in range(len(captcha_array_filtered)):
                captcha_string += captcha_array_filtered[captcha_letter][0]
                
            return captcha_string

def get_statics(path):
  total_tested = 0
  total_hitted = 0

  for arq in glob.glob(path):
    expected = arq[len(path)-1:-4]
    result_from_arq = Captcha_detection(arq)
    total_tested += 1
    print("(Expected: " + expected + ", result: " + result_from_arq + ")\n")
    if(expected == result_from_arq):
      total_hitted += 1
      
  estatistica = float(total_hitted)/float(total_tested)
  print("Quantidade testada: " + str(total_tested) + ", n_acertos: " + str(total_hitted) + "\n")
  print("Estatística de acerto: %f " %estatistica)  

  
PATH_IMAGE = sys.argv[1]
if(PATH_IMAGE != "estatistica"):
  print("\n\nResultado do arquivo " + PATH_IMAGE + " -> " + Captcha_detection(PATH_IMAGE) + "\n") 
else:
  get_statics(sys.argv[2])
