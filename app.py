import os
import cv2
import imutils
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


filepath = './face_detector/'
model_path = './model/mask_detectors.h5'
model_video = './model/mask_detectors.model'
image_test = './images_test/'
MY_CONFIDENCE = 0.9
BATCH_SIZE = 32
IMG_SIZE = (160, 160)


def local_css(file_name):
    """ Method for reading styles.css and applying necessary changes to HTML"""
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def get_images_with_faces():
    global face_mask
    print('[INFO] loading face detector model...')
    # face detector
    prototxtPath = os.path.sep.join([filepath, 'deploy.prototxt'])
    weightsPath = os.path.sep.join([filepath, 'res10_300x300_ssd_iter_140000.caffemodel'])
    face_model = cv2.dnn.readNet(prototxtPath, weightsPath)

    # facemask detector model (trained in other notebook)
    print('[INFO] loading face mask detector model...')
    model = load_model(model_path)

    # load the input image from disk and grab the image spatial
    # dimensions
    image = cv2.imread('./images_test/test.jpg')
    assert not isinstance(image, type(None)), 'image not found'
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # construct a blob from the image
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
                label = 'With Mask' if results[0][0] < 0 else 'Without Mask'
                color = (255, 165, 0) if label == "With Mask" else (0, 0, 255)
                cv2.putText(image, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
                face_mask = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


get_images_with_faces()


def get_video_faces(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence

        if confidence > MY_CONFIDENCE:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)


# load our serialized face detector model from disk
prototxtPath = os.path.sep.join([filepath, "deploy.prototxt"])
weightsPath = os.path.sep.join([filepath, "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model(model_video)


class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        print('getting frame')
        frame = frame.to_ndarray(format="bgr24")
        frame = imutils.resize(frame, width=400)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = get_video_faces(frame, faceNet, maskNet)
        print('predicting face mask')

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            print('loading predictions')

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "With Mask" if mask > withoutMask else "Without Mask"
            color = (255, 165, 0) if label == "With Mask" else (0, 0, 255)
            print('printing the mask detections')

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            print('finishing process')

        return frame


def mask_detection():
    local_css('css/styles.css')
    st.markdown('<h1 align="center">😷 Face Mask Detection</h1>', unsafe_allow_html=True)
    activities = ['Image', 'Webcam']
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.sidebar.markdown('# Are they wearing a mask?')
    choice = st.sidebar.selectbox('Choose among the given options:', activities)
    
    if choice == 'Image':
         
        st.markdown('<h2 align="center">Detection on Image :camera: </h2>', unsafe_allow_html=True)
        st.markdown('### Upload your image here ⬇')
        image_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'])  # upload image
        if image_file is not None:
            our_image = Image.open(image_file)  # making compatible to PIL
            im = our_image.save('./images_test/test.jpg')
            saved_image = st.image(image_file, caption='', use_column_width=True)
            st.markdown('<h3 align="center">Image uploaded successfully!</h3>', unsafe_allow_html=True)
            if st.button('Facemask detector is analysing the image'):
                st.image(face_mask, use_column_width=True)
        else:
            st.markdown('<h3 align="center">Image limit is 200MB</h3>', unsafe_allow_html=True)

    if choice == 'Webcam':
        st.markdown('<h2 align="center">Detection on Webcam</h2>', unsafe_allow_html=True)
        webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)


mask_detection()
print('process finished')
