# Example 4.14 - Webcam classification with VGG16/ResNet50/MobileNet/InceptionV3
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications import vgg16, resnet50, mobilenet, inception_v3

# Select one model and matching preprocessing function.
#model = vgg16.VGG16(weights='imagenet'); preprocess = vgg16.preprocess_input; image_size = 224
model = resnet50.ResNet50(weights='imagenet'); preprocess = resnet50.preprocess_input; image_size = 224
#model = mobilenet.MobileNet(weights='imagenet'); preprocess = mobilenet.preprocess_input; image_size = 224
#model = inception_v3.InceptionV3(weights='imagenet'); preprocess = inception_v3.preprocess_input; image_size = 299

print(model.summary())
camera = cv2.VideoCapture(0)
while camera.isOpened():
    ok, cam_frame = camera.read()
    if not ok:
        break
    frame = cv2.resize(cam_frame, (image_size, image_size))
    image_batch = np.expand_dims(img_to_array(frame), axis=0)
    processed_image = preprocess(image_batch.copy())
    predictions = model.predict(processed_image, verbose=0)
    label = decode_predictions(predictions)
    text = '{}, {:.1f}'.format(label[0][0][1], label[0][0][2])
    cv2.putText(cam_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('video image', cam_frame)
    if cv2.waitKey(30) == 27:
        break
camera.release()
cv2.destroyAllWindows()
