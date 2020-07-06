import os
import time
import numpy as np
import cv2

from flask import Flask, render_template, Response
from tensorflow.python.platform import gfile
# force tensorflow to work like 1.x version
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from utils import load_graph, load_class_names, encode_img, \
				  output_boxes, non_max_suppression, draw_outputs
				  
# Setting up hyperparameters
MODEL_SIZE = (416, 416,3)
MAX_OUTPUT_SIZE = 5				#Limit max amount of bboxes
MAX_OUTPUT_SIZE_PER_CLASS = 20 
IOU_THRESHOLD = 0.4 
CONFIDENCE_THRESHOLD = 0.3 

# Set model files pathes
PB_PATH = "./model_files/frozen_darknet_yolov3_model.pb"
CLASS_NAMES_PATH = './model_files/obj.names'

# Load tf model from .pb file
graph = load_graph(PB_PATH)
class_names = load_class_names(CLASS_NAMES_PATH)
print('Model and classes loaded!')
print('Class names', class_names)

# Print list of graph nodes. Useful for input/output selection.
# ~ for op in graph.get_operations():
	# ~ print(op.name)
	
# Set input and output nodes. You can find them in list of graph nodes.
x = graph.get_tensor_by_name('inputs:0')
y = graph.get_tensor_by_name('output_boxes:0')

	
def write_predict(raw_image, graph, fps):
	with tf.Session(graph=graph) as sess:
		# Encode test image
		raw_img, test_input = encode_img(raw_image, MODEL_SIZE)
		print('test_input shape', test_input.shape)
		# Run tf model
		pred = sess.run(y, feed_dict={x: test_input} )
		# Handle model output
		boxes, scores, classes, nums = output_boxes( \
			pred, MODEL_SIZE,
			max_output_size=MAX_OUTPUT_SIZE,
			max_output_size_per_class=MAX_OUTPUT_SIZE_PER_CLASS,
			iou_threshold=IOU_THRESHOLD,
			confidence_threshold=CONFIDENCE_THRESHOLD)	   
		img = draw_outputs(raw_img, boxes, scores, classes, nums, class_names)
		# Add fps value
		words_color = (0,0,255) #BGR
		if fps is not None:
			cv2.putText(img, "FPS: {:.2f}".format(fps), (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, words_color, 1)
		# Write final result
		cv2.imwrite('result.jpg', img)
		print('scores', scores.eval())	
		
#DELETE AFTER DEPLOYING	
# ~ # Sample predict test
# ~ sample_img_path = './sample_images/crowd.jpg'  
# ~ write_predict(sample_img_path, graph)

def gen_from_cam():
	# We will limit fps, beause of my weak cpu
	# https://stackoverflow.com/questions/52068277/change-frame-rate-in-opencv-3-4-2
	iterationLimit = 0  # Should be less than that iteration time and adapt to this value in while loop. 
	startTime = time.time()
	fps = 0
	
	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		raise IOError("Can't open webcam")
	while True:
		ret, frame = cap.read()
		nowTime = time.time()
		if (float(nowTime - startTime)) > iterationLimit:
			
			# Do YOLO stuff on the flipped frame
			frame = cv2.flip(frame, 1)
			write_predict(frame, graph, fps)
				
			# Reread image and push it to the screen
			f = open("result.jpg", 'rb').read()
			startTime = time.time()
			
			# Update time parameters
			iterationTime = startTime - nowTime
			fps = 1/iterationTime 
			iterationLimit = np.round(iterationTime, 2) 
			print('Iteration time: {}, Iteration limit: {}'. format(iterationTime, iterationLimit))
		try:
			yield (b'--frame\r\n'
				   b'Content-Type: image/jpeg\r\n\r\n' + f + b'\r\n')
		# Due to fps limitation we could get error "refference f before assignment".
		except:
			continue 

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

				
@app.route('/video_feed')
def video_feed():
	return Response(gen_from_cam(),
					mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
	app.run(host='0.0.0.0', debug=False)
    
    
    





