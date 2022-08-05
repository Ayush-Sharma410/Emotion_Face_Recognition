import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import matplotlib.pyplot as plt
import pandas as pd
import speech_recognition as sr
import pyttsx3
from matplotlib import pyplot, pylab
import time
import smtplib
server=smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login('your email', 'password')

starting_time = time.time()
ending_time = time.time()

engine=pyttsx3.init()


face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
classifier =load_model(r'model.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)


c_happy=0
c_sad=0
c_disgust=0
c_fear=0
c_neutral=0
c_surprise=0
c_angry=0
c_frames=0

path = "test_images"
images = []
classNames = []
mylist = os.listdir(path)
#print(mylist)
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
    #print(classNames)

def findencodings(images):
    encodeList = []
    for img in images:
         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
         encode = face_recognition.face_encodings(img)[0]
         encodeList.append(encode)
    return encodeList
def markAttendence(name):
    with open("emotion_analysis.csv",'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry =line.split(',')
            nameList.append(entry[0])
        if name in nameList:
            now = datetime.now()
            dtstring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring},{label}')

cap = cv2.VideoCapture(0)
encodelistknown = findencodings(images)
print("encoding complete")

while True :
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    _, frame = cap.read()
    c_frames += 1
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    if (ending_time - starting_time) <= 36:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]

                if label == "Happy":
                    c_happy += 1
                elif label == "Sad":
                    c_sad += 1
                elif label == "Surprise":
                    c_surprise += 1
                elif label == "Disgust":
                    c_disgust += 1
                elif label == "Fear":
                    c_fear += 1
                elif label == "Neutral":
                    c_neutral += 1
                elif label == "Angry":
                    c_angry += 1

                label_position = (x, y)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Emotion Detector', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        for encodeFace, faceloc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodelistknown, encodeFace)
            faceDis = face_recognition.face_distance(encodelistknown, encodeFace)
            print(faceDis)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print(name)
                y1, x2, y2, x1 = faceloc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
                markAttendence(name)

            cv2.imshow("webcam", img)

            cv2.waitKey(1)
            ending_time = time.time()
    else:
        break


print(c_happy / c_frames)

if c_happy>max(c_sad,c_fear,c_angry,c_disgust,c_neutral,c_surprise):
    engine.say('Hello'+name)
    engine.say('I am here to tell your expression')
    engine.say('You are so happy')
    engine.runAndWait()
    server.sendmail('your email', 'your email', name+' is happy')
elif c_sad>max(c_happy,c_fear,c_angry,c_disgust,c_neutral,c_surprise):
    engine.say('Hello  '+name)
    engine.say('I am here to tell your expression')
    engine.say('You are so sad')
    engine.runAndWait()
    server.sendmail('your email', 'your email', 'You are sad')
elif c_fear>max(c_sad,c_happy,c_angry,c_disgust,c_neutral,c_surprise):
    engine.say('Hello  '+name)
    engine.say('I am here to tell your expression')
    engine.say('You are so frightened')
    engine.runAndWait()
    server.sendmail('your email', 'your email', 'You are frightened')
elif c_angry>max(c_sad,c_fear,c_happy,c_disgust,c_neutral,c_surprise):
    engine.say('Hello '+name)
    engine.say('I am here to tell your expression')
    engine.say('You are so angry')
    engine.runAndWait()
    server.sendmail('your email', 'your email', 'You are angry')
elif c_disgust>max(c_sad,c_fear,c_angry,c_happy,c_neutral,c_surprise):
    engine.say('Hello '+name)
    engine.say('I am here to tell your expression')
    engine.say('You are so disgusted')
    engine.runAndWait()
    server.sendmail('', '', 'You are being disgust')
elif c_neutral>max(c_sad,c_fear,c_angry,c_disgust,c_happy,c_surprise):
    engine.say('Hello'+name)
    engine.say('I am here to tell your expression')
    engine.say('Your expression is neutral')
    engine.runAndWait()
    server.sendmail('your email', 'your email', 'Your expression is neutral')
elif c_surprise>max(c_sad,c_fear,c_angry,c_disgust,c_neutral,c_happy):
    engine.say('Hello   '+name)
    engine.say('I am here to tell your expression')
    engine.say('You are so surprised')
    engine.runAndWait()
    server.sendmail('your email', 'your email', 'You are surprised')
plt.style.use('bmh')
df = pd.read_csv('emotion_analysis.csv')
x = df['Time']
y = df['Emotion']
plt.xlabel('Time')
plt.ylabel('Emotion')
plt.scatter(x, y)
plt.bar(x, y)
plt.show()
cap.release()
cv2.destroyAllWindows()
