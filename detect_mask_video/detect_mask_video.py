# python detect_mask_video.py

# import the necessary packages
from cgitb import text
from tracemalloc import start
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from tensorflow.keras.applications.efficientnet import preprocess_input
# from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
	# tạo blob phù hợp vs Caffe model face detector
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# phát hiện các khuôn mặt trong hình ảnh bằng cách truyền blod vào facenet
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# khởi tạo list khuôn mặt [faces] và vị trí tương ứng của chúng[locs]
	#list các dự đoán [preds]
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the detection
		confidence = detections[0, 0, i, 2]
		if confidence > args["confidence"]:
			# tính toán tọa độ bounding box cho object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			#đảm bảo các bounding box nằm trong kích thước của frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# trích xuất face Roi và chuyển nó từ BGR sang RGB
			# tiền xử lý, thay đổi kích thước thành 224x224
			face = frame[startY:endY, startX:endX]
			if face.any():
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)

				# add the face and bounding boxes to their respective
				# lists
				faces.append(face)
				locs.append((startX, startY, endX, endY))

	# chỉ đưa ra dự đoán nếu có ít nhất 1 khuôn mặt được phát hiện
	if len(faces) > 0:
		# để suy luận nhanh hơn thì sẽ đưa ra dự đoán hàng loạt về tất cả các khuôn mặt
		# cùng 1 lúc 
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	#trả về vị trí locs và kết quả dự đoán preds
	return (locs, preds)

# tạo các đối số và đường dẫn
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="face_mask_detect.model",
	# default="face_mask_detect_efficientnetbo.model",
	# default="face_mask_detect_resnet50.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# sử dụng Caffe model để định vị các khuôn mặt trong ảnh
print("[INFO] loading face mask detect model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load model nhận diện khẩu trang đúng cách
print("[INFO] loading face mask detect model...")
maskNet = load_model(args["model"])

# read video stream
print("[INFO] starting video stream...")
vcap = cv2.VideoCapture("rtsp://192.168.43.218:8554/webCamStream")


# loop over the frames from the video stream
while True:
	# lấy frame và thay đổi kích thước với chiều rộng tối đa là 400 pixel
	
	_, frame = vcap.read()
	frame = imutils.resize(frame, width=400)
	# xác định các khuôn mặt trong frame và xác định xem có đeo khẩu trang hay không
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# dùng vòng lặp phát hiện các vị trí khuôn mặt được phát hiện và các vị trí tương ứng 
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(incorrect_mask,mask, no_mask) = pred
	
		# xác định label và color sử dụng để vẽ  bounding box và text
		temp = np.array([mask, no_mask, incorrect_mask])
		_labels = ['Mask', 'No Mask', 'Incorrect Mask']
		_color =  [(0,255,0), (0,0,255), (255,0,0)]
		
		_previous = ('No Mask')
		index_max = np.argmax(temp)
		if np.max(temp) > 0.8:
			label = _labels[index_max]
		else:
			label = _previous
		_previous = label
		color = _color[index_max]
		# hiển thị label và độ chính xác
		text = "{}: {:.2f}%".format(label, max(temp) * 100)
		
		# hiển thị nhãn và bounding  box 
		cv2.putText(frame, text, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# hiển thị output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# nhấn q để kết thúc vòng lặp
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vcap.stop()
vs.stop()