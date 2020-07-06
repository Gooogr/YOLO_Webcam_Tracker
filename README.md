## Object detection web-page based on flask, opencv and tiny YOLO


### Convert weights
copy weights and names files into ./tensorflow-yolo-v3/model_files
```
cd tensorflow-yolo-v3 &&
python3 convert_weights_pb.py \
--class_names ./model_files/obj.names \
--weights_file ./model_files/yolov3-tiny-face-palms.weights \
--data_format NHWC \
--tiny
```

### Run
```python app.py```

### Acknowledgments
The simplest implementation of the flask webcam streamer: [streaming_webcam_with_flask](https://github.com/py-chemist/streaming_webcam_with_flask)<br>
YOLOv3 weights converter: [tensorflow-yolo-v3](https://github.com/mystic123/tensorflow-yolo-v3)<br>
Custom TensorFlow binaries for old CPU without AVX instruction suppot: [TensorFlow binaries supporting AVX, FMA, SSE](https://github.com/lakshayg/tensorflow-build)<br>
TensorFlow YOLOV3/TinyYOLOV3 implementation with tutirials: [YOLOv3_TF2](https://github.com/RahmadSadli/Deep-Learning/tree/master/YOLOv3_TF2)
