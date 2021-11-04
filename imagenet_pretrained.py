# import các gói thư viện cần thiết
from keras.applications.resnet import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse
import cv2
from imutils import paths
from datasets import simpledatasetloader
from preprocessing import imagetoarraypreprocessor
from preprocessing import simplepreprocessor

# Cách dùng: python imagenet_pretrained.py -i <path/tên file> [-m vgg16]

# Xây dựng cấu trúc để sử dụng bằng dòng lệnh.
# Lưu ý: model vgg16 là mặc định, nghĩa là không có tham số -m <mô hình>
# Thì mặc định dùng vgg16
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,help="Đường dẫn ảnh đầu vào để dự đoán")
ap.add_argument("-m", "--model", type=str, default="vgg16",help="Tên của model pre-trained")
args = vars(ap.parse_args())

# Định nghĩa từ điển chứa tên các model
MODELS = {
            "vgg16": VGG16,
            "vgg19": VGG19,
            "inception": InceptionV3,
            "xception": Xception,
            "resnet": ResNet50
          }

# Đảm bảo sử dụng tên model là tham số của dòng lệnh
if args["model"] not in MODELS.keys():
    raise AssertionError("Tên model không có trong từ điển")

# Khởi tạo ảnh đầu vào có kích thước (224x224) along with
inputShape = (224, 224)
preprocess = imagenet_utils.preprocess_input

# Nếu dùng the InceptionV3 hoặc Xception thì kích thước ảnh đầu vào (299x299)
if args["model"] in ("inception", "xception"):
    inputShape = (299, 299)
    preprocess = preprocess_input

# Nạp trọng số mạng (network weights)
# (Chú ý: tại thời điểm chạy lần đầu của một mạng nào đó thì các trọng số sẽ được
# down về. Tùy thuộc mạng mà dung lượng các trọng số từ 90-575MB, trọng số sẽ được
# lưu vào bộ nhớ đệm và các lần chạy tiếp theo sẽ nhanh hơn nhiều)
print("[INFO] Đang nạp mô hình {} ...".format(args["model"]))
Network = MODELS[args["model"]]
model = Network(weights="imagenet")

print("[INFO] đang nạp và tiền xử lý ảnh ...")
imagePaths = np.array(list(paths.list_images(args["image"])))
sp = simplepreprocessor.SimplePreprocessor(32, 32) # Thiết lập kích thước ảnh 32 x 32
iap = imagetoarraypreprocessor.ImageToArrayPreprocessor() # Gọi hàm để chuyển ảnh sang mảng
sdl = simpledatasetloader.SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths)
data = data.astype("float") / 255.0


for (i, imagePath) in enumerate(imagePaths):
    image = load_img(imagePath, target_size=inputShape)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess(image)

    # Dự đoán và phân lớp ảnh
    print("[INFO] Phân lớp ảnh bằng mô hình '{}'...".format(args["model"]))
    preds = model.predict(image)
    P = imagenet_utils.decode_predictions(preds)  # Trả về danh sách các tham số dự đoán P

    # Lấy và hiển thị một số thông tin về kết quả dự đoán ảnh:
    # - imagenetID: ID của ảnh trong imagenet
    # - label: Nhãn của ảnh
    # - prob: Phần trăm (xác suất) dự đoán
    for (i, (imagenetID, label, prob)) in enumerate(P[0]):
        print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))

    # Sau khi đã dự đoán --> Hiển thị ảnh và vẽ kết quả phân loại trên ảnh
    img = cv2.imread(imagePath)
    (imagenetID, label, prob) = P[0][0]
    cv2.putText(img, "Label: {}".format(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.imshow("Ket qua phan lop", img)
    cv2.waitKey(0)

