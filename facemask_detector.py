import os
from os import walk
from shutil import copyfile
import cv2
import numpy as np
from tensorflow.keras.models import load_model

filepath = './face_detector/'
model_path = './model_test/mask_detector.h5'
MY_CONFIDENCE = 0.9
BATCH_SIZE = 32
IMG_SIZE = (160, 160)

print('starting the final project')


def get_images_with_faces():
    # face detector
    prototxtPath = os.path.sep.join([filepath, 'deploy.prototxt'])
    weightsPath = os.path.sep.join([filepath, 'res10_300x300_ssd_iter_140000.caffemodel'])
    face_model = cv2.dnn.readNet(prototxtPath, weightsPath)

    # facemask detector model (trained in other notebook)
    model = load_model(model_path)

    # loading images
    mask_images = './dataset/with_mask/'
    without_images = './dataset/without_mask/'
    copy_images = './images/copy/'
    f = []

    try:
        os.mkdir(copy_images)
    except OSError:
        print('File exists: continue')

    for (dirpath, dirnames, filenames) in walk(mask_images):
        for file in filenames:
            copyfile(mask_images + file, copy_images + 'mask_' + file)
            f.append(copy_images + 'mask_' + file)
        break

    for (dirpath, dirnames, filenames) in walk(without_images):
        for file in filenames:
            copyfile(without_images + file, copy_images + 'without_mask_' + file)
            f.append(copy_images + 'without_mask_' + file)
        break

    for img in f:
        image = cv2.imread(img)
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

        # detecting faces in images
        print('computing face detections...')
        face_model.setInput(blob)
        detections = face_model.forward()

        # detecting mask/without mask in every face
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence >= MY_CONFIDENCE:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                face = image[startY:endY, startX:endX]

                if len(face) != 0:
                    face = cv2.resize(face, IMG_SIZE)
                    face = face[np.newaxis, ...]

                    # predict mask/without mask with the model
                    print('predicting the results')
                    results = model.predict_on_batch(face)
                    # print results
                    label = 'Mask' if results[0][0] < 0 else 'No Mask'
                    color = (0, 255, 0) if label == 'Mask' else (0, 0, 255)
                    cv2.putText(image, label, (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
        cv2.imshow('Output', image)
        cv2.waitKey(0)


if __name__ == "__main__":
    get_images_with_faces()
    print('process finished')
