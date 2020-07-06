'''
Based on:
https://docs.opencv.org/3.4.2/d7/d8b/tutorial_py_face_detection.html
Example of Haar-cascade Detection in OpenCV

Connect to the webcam and bounding your face and eyes.
'''

import numpy as np
import cv2 
import os

DIR_PATH = '/home/grigoriy/envs/face_touch/lib/python3.6/site-packages/cv2/data/'

face_cascade = cv2.CascadeClassifier(os.path.join(DIR_PATH,'haarcascade_frontalface_default.xml'))
eye_cascade = cv2.CascadeClassifier(os.path.join(DIR_PATH,'haarcascade_eye.xml'))

capture = cv2.VideoCapture(0)
while capture.isOpened():
		pressed_key = cv2.waitKey(1)
		_, frame = capture.read()
		frame = cv2.flip(frame, 1)
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
		for (x,y,w,h) in faces:
			cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
			# Eyes detection part
			roi_gray = gray_frame[y:y+h, x:x+w]
			roi_color = frame[y:y+h, x:x+w]
			eyes = eye_cascade.detectMultiScale(roi_gray)
			for (ex,ey,ew,eh) in eyes:
				cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
		cv2.imshow("Live Feed", frame)




