# python detect_mask_image.py --image images/pic1.jpeg

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os
def mask_image():
	# tạo các đối số và đường dẫn 
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True,
		help="path to input image")
	ap.add_argument("-f", "--face", type=str,
		default="face_detector",
		help="path to face detector model directory")
	ap.add_argument("-m", "--model", type=str,
		default="face_mask_detect.model",
		help="path to trained face mask detector model")
	ap.add_argument("-c", "--confidence", type=float, default=0.5,
		help="minimum probability to filter weak detections")
	args = vars(ap.parse_args())

	# sử dụng Caffe model để định vị các khuôn mặt trong ảnh
	print("[INFO] loading face detector model...")
	prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
	weightsPath = os.path.sep.join([args["face"],
		"res10_300x300_ssd_iter_140000.caffemodel"])
	net = cv2.dnn.readNet(prototxtPath, weightsPath)

	# load model phát hiện đeo khẩu trang đúng cách
	print("[INFO] loading face mask detector model...")
	model = load_model(args["model"])

	# load the input image from disk, clone it, and grab the image spatial
	# dimensions
	image = cv2.imread(args["image"])
	orig = image.copy()
	(h, w) = image.shape[:2]

	# tạo blob phù hợp vs Caffe model face detector
	blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# phát hiện các khuôn mặt trong hình ảnh bằng cách truyền blod vào facenet
	print("[INFO] computing face detections...")
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]
		if confidence > args["confidence"]:
			# tính toán tọa độ bounding box cho object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			#đảm bảo các bounding box nằm trong kích thước của frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# trích xuất face ROI và xử lý ảnh đầu vào
			face = image[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)

			# phát hiện đeo khẩu trang đúng cách
			(incorrect_mask,mask, no_mask ) = model.predict(face)[0]

			#xác định label và color sử dụng để vẽ  bounding box và text
			temp= np.array([mask, no_mask, incorrect_mask])
			_labels = ['Mask', 'No Mask', 'Incorrect Mask']
			_color =  [(0,255,0), (0,0,255), (255,0,0)]
			index_max = np.argmax(temp)
			_previous = ('No Mask')
			if np.max(temp) > 0.3:
				label = _labels[index_max]
			else:
				label = _previous
			_previous = label
			color = _color[index_max]

			# hiển thị label và độ chính xác
			text = "{}: {:.2f}%".format(label, max(temp) * 100)

			# hiển thị nhãn và bounding  box
			cv2.putText(image, text, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

	# show the output image
	cv2.imshow("Frame", image)
	cv2.waitKey(0)
	
if __name__ == "__main__":
	mask_image()
