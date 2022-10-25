## python train_mask_detector.py --dataset dataset

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.applications.efficientnet import EfficientNetB0
# from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from tensorflow.keras.applications.efficientnet import preprocess_input
# from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
# from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# tạo argument đường dẫn
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
    help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str,
    default="face_mask_detect.model",
    # default="face_mask_detect_efficientnetbo.model",
    # default="face_mask_detect_resnet50.model",
    help="path to output face mask detector model")
args = vars(ap.parse_args())

#khởi tạo hyperparameters
INIT_LR = 1e-4
EPOCHS = 20
BS = 4

# tạo các list data, labels từ dataset
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# make datapath
for imagePath in imagePaths:
    # gán label theo tên của ảnh ,
    # dataset/incorrect_mask/01002_Mask_mounth.jpg
    label = imagePath.split(os.path.sep)[-2]

    # tiền xử lý ảnh để phù hợp đầ vào mô hình
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image) 

    data.append(image)
    labels.append(label)

# convert data và labels về numpy
data = np.array(data, dtype="float32")
labels = np.array(labels)
# biểu diễn labels dạng one hot vector
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# tạo data loader,chia tập train và test tỷ lệ 80:20 và chia ảnh với việc chọn random_state=42
(trainX, testX, trainY, testY) = train_test_split(data, labels,
    test_size=0.20, stratify=labels, random_state=42)

# tăng cường dữ liệu huấn luyện
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# backbone MobileNetV2 
baseModel = MobileNetV2(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(224, 224, 3)))
# baseModel = EfficientNetB0(weights="imagenet", include_top=False,
    # input_tensor=Input(shape=(224, 224, 3)))

# baseModel = ResNet50(weights="imagenet", include_top=False,
#   input_tensor=Input(shape=(224, 224, 3)))

# head CNN
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(3, activation="softmax")(headModel)

#model=backbone+head
model = Model(inputs=baseModel.input, outputs=headModel)
model.summary()
# đóng băng tất cả các params trong backbone để tránh update
for layer in baseModel.layers:
    layer.trainable = False

# compile model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])

# train the head of the network
print("[INFO] training head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)

#đưa ra dự đoán hay kiểm tra tập val
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

#với mỗi ảnh thì đưa ra label có xác suất dự đoán là lớn nhất
predIdxs = np.argmax(predIdxs, axis=1)

# in ra báo cáo thể hiện chỉ số phân loại 
print(classification_report(testY.argmax(axis=1), predIdxs,
    target_names=lb.classes_))

# save model 
print("[INFO] saving mask detector model...")
model.save(args["model"], save_format="h5")

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
