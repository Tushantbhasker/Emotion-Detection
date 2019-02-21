import cv2
import numpy as np
from keras.models import load_model
import os
import sys
# Specifing our model loction
sys.path.append(os.path.abspath("./model"))

from load import *

global model, graph
#initialize these variables
model, graph = init()
# print(model,graph)


# laptop camera
cap = cv2.VideoCapture(0)


# pre - trinaed xml file for detecting faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX


# loading saved cnn model
# model = load_model('model.h5')

feelings_faces = []
for emotion in range(7):
    feelings_faces.append(cv2.imread('./emojis/' + (str)(emotion) + '.png', -1))

# predicting face emotion using saved model
def get_emo(im):
    im = im[np.newaxis, :, :, np.newaxis]
    res = model.predict_classes(im,verbose=0)
    emo = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    return emo[res[0]], res[0]

def recognize_face(im):
    im = cv2.resize(im, (48, 48))
    return get_emo(im)


while True:
    _, frame = cap.read()
    flip_fr = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        fc = frame[y:y+h, x:x+w, :]
        gfc = cv2.cvtColor(fc, cv2.COLOR_BGR2GRAY)
        out, idx = recognize_face(gfc)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        flip_fr = cv2.flip(frame,1)
        cv2.putText(flip_fr, out, (30, 30), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', feelings_faces[idx])
        # face_image = feelings_faces[idx]
        # frame[x-120:x, y-120:y, :] = face_image[:, :, 0:3]

        # face_image = feelings_faces[idx]
        # for c in range(0,3):
        #   frame[x-120:x, y-120:y, c] = face_image[:, :, c]
            
    
    cv2.imshow('rgb', flip_fr)

    # press esc to close the window
    k = cv2.waitKey(1) & 0xEFFFFF
    if k==27:   
        break
    elif k==-1:
        continue
    else:
        continue

cv2.destroyAllWindows()