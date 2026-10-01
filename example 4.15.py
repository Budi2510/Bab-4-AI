# Example 4.15 - Webcam classification with image-classifiers
# Install first: pip install image-classifiers
import cv2
import numpy as np
from keras.applications.imagenet_utils import decode_predictions
from classification_models.keras import Classifiers

# Choose a model: vgg16, resnet50, mobilenetv2, densenet201, inceptionv3
clf, preprocess_input = Classifiers.get('vgg16')
image_size = 224
# clf, preprocess_input = Classifiers.get('inceptionv3')
# image_size = 299

model = clf(input_shape=(image_size, image_size, 3), weights='imagenet', classes=1000)
model.summary()
camera = cv2.VideoCapture(0)
while camera.isOpened():
    ret, cam_frame = camera.read()
    if not ret:
        break
    frame = cv2.resize(cam_frame, (image_size, image_size))
    image = np.expand_dims(np.asarray(frame), 0)
    image = preprocess_input(image)
    preds = model.predict(image, verbose=0)
    label = decode_predictions(preds)
    text = '{}, {:.1f}'.format(label[0][0][1], label[0][0][2])
    cv2.putText(cam_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow('Classification', cam_frame)
    if cv2.waitKey(30) == 27:
        break
camera.release()
cv2.destroyAllWindows()
