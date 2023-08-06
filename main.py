import speech_recognition as sr
from time import sleep
import pyttsx3
r = sr.Recognizer()
mic=sr.Microphone()
engine=pyttsx3.init()
voices=engine.getProperty('voices')
engine.setProperty('voice',voices[1].id)
def talk(text):
    engine.say(text)
    engine.runAndWait()
import face_recognition as fr
import os
import cv2
import face_recognition
import numpy as np
from time import sleep
import imp
import time
import io,requests
import pytesseract
pytesseract.pytesseract.tesseract_cmd='C:/Program Files/Tesseract-OCR/tesseract'
import re
from utils import *

def get_encoded_faces():
    encoded={}
    for dirpath,dnames,fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face=fr.load_image_file("faces/"+f)
                encoding=fr.face_encodings(face)[0]
                encoded[f.split(".")[0]]=encoding
    return encoded

def unknown_image_encoded(img):
    face=fr.load_image_file("faces/"+img)
    encoding= fr.face_encodings(face)[0]
    return encoding

def clasify_face(im):
    faces=get_encoded_faces()
    faces_encoded=list(faces.values())
    known_face_names=list(faces.keys())
    img=cv2.imread(im,1)
    face_locations=face_recognition.face_locations(img)
    unknown_face_encodings=face_recognition.face_encodings(img,face_locations)
    face_names=[]
    for face_encoding in unknown_face_encodings:
        matches=face_recognition.compare_faces(faces_encoded,face_encoding)
        name="Unknown"
        face_distances=face_recognition.face_distance(faces_encoded,face_encoding)
        best_match_index=np.argmin(face_distances)
        if matches[best_match_index]:
            name=known_face_names[best_match_index]
            print(name)
        face_names.append(name)
        return face_names

def take_image(string):
    words=string.split()
    print(words[-1])
    cam=cv2.VideoCapture(0)
    cv2.namedWindow("test")
    img_counter=0
    ret,frame=cam.read()
    if not ret:
        print("failed to grab frame")
        return
    cv2.imshow("test",frame)
    k=cv2.waitKey(1)
    img_name=words[-1]+".png".format(img_counter)
    path='D:\Vision-Guide\faces'
    cv2.imwrite(os.path.join(path,img_name),frame)
    print("{} written!".format(img_name))
    img_counter+=1
    if os.path.exists(os.path.join("absolute path",img_name)):
        os.remove(os.path.join("absolute path",img_name))
    else:
        print("The file does not exist")
    return

def check_image():
    cam=cv2.VideoCapture(0)
    cv2.namedWindow("test")
    img_counter=0
    ret, frame=cam.read()
    if not ret:
        print("failed to grab frame")
        return
    cv2.imshow("test",frame)
    k=cv2.waitKey(1)
    img_name="test.jpg".format(img_counter)
    cv2.imwrite(img_name,frame)
    print("{} written!".format(img_name))
    img_counter+=1
    if os.path.exists(os.path.join("absolute path",img_name)):
        os.remove(os.path.join("absolute path",img_name))
    else:
        print("The file does not exist")
    name=clasify_face("test.jpg")
    return name

def check_surrounding():
    net=cv2.dnn.readNet("dnn_model/yolov4-tiny.weights","dnn_model/yolov4-tiny.cfg")
    model=cv2.dnn_DetetionModel(net)
    model.setInputParams(size=(320,320),scale=1/255)
    classes=[]
    with open("dnn_model/classes.txt","r") as file_object:
        for class_name in file_object.readlines():
            class_name=class_name.strip()
            classes.append(class_name)
    cap=cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,720)
    ret,frame=cap.read()
    (class_ids,scores,bboxes)=model.detect(frame,confThreshold=0.3,nmsThreshold=.4)
    obj_names=[]
    for class_id,score,bbox in zip(class_ids,scores,bboxes):
        (x,y,w,h)=bbox
        class_name=classes[class_id]
        print(class_name)
        obj_names.append(class_name)
    return obj_names 



def read():
    cam=cv2.VideoCapture(0)
    cv2.namedwindow("test")
    img_conter=0
    ret,frame=cam.read()
    if not ret:
        print("failed to grab frame")
        return
    cv2.imshow("test",frame)

    k=cv2.waitKey(1)
    img_name="read.jpg".format(img_counter)
    cv2.imwrite(img_name,frame)
    print("{} wriiten!".format(img_name))
    img_counter+=1
    if os.path.exists(os.path.join("absolute path",img_name)):
        os.remove(os.path.join("absolute path",img_name))
    else:
        print("The file does not exist")
img1 ="read.jpg"
img =cv2.imread(img1)
_, compressedimg=cv2.imencode(".jpg",img)
file_bytes=io.BytesIO(compressedimg)
url_api="https://api.ocr.space/parse"
response =requests.post(url_api,
        files={img1 :file_bytes},
        data ={"apikey":"K85285375488957"})
result =response.json()
output =result["ParsedResults"][0]["ParsedText"]
print(output)


def bill():
    cam=cv2.VideoCapture(0)
    cv2.namedWindow("test")
    img_counter =0
    ret,frame=cam.read()
    if not ret:
        print("failed to grab frame")
        return
    cv2.imshow("test",frame)

    k=cv2.waitKey(1)


    img_name="bill.png".format(img_counter)
    cv2.imwrite(img_name, frame)
    print("{} written!".format(img_name))
    img_counter+= 1

    if os.path.exists(os.path.join("absolute path",img_name)):
        os.remove(os.path.join("absolute path", img_name))
    else:
       print("The file does not exist")
    image =cv2.imread("bill.png",0)
    text =(pytesseract.image_to_string(image)).lower()

    match=re.findall(r'\d+[/.-]\d+[/.-]\d+',text)

    st=" "
    st=st.join(match)
    print(st)
    talk("Bill on the date of")
    talk(st)

    price=re.findall(r'[\$\£\€](\d+(?:\.\d{1,2})?)',text)
    price =list(map(float, price))
    print(max(price))
    x=max(price)
    talk("price is")
    talk(x)
    print(output)
    return output


def currency():
    cam =cv2.VideoCapture(0)
    cv2.nameWindow("test")
    img_counter=0
    ret, frame=cam.read()
    if not ret:
        print("failed to grab frame")
        return
    cv2.imshow("test",frame)

    k=cv2.waitKey(1)

    img_name="currency.png".format(img_counter)

    cv2.imwrite(img_name, frame)
    print("{} written!",format(img_name))
    img_counter+=1


    if os.path.exists(os.path.join("absolute path",img_name)):
        os.remove(os.path.join("absolute path",img_name))
    else:
        print("The file does not exist")
    max_val=8
    max_pt=-1
    max_kp=0

    orb=cv2.ORB_create()
    test_img = read_img('currency.png')   
    original =resize_img(test_img,0.4)

    (kp1, des1)=orb.detectAndCompute(test_img,None)
    training_set=['files/20.jpg','files/50.jpg','files/100.jpg','files/500.jpg','files/2000']

    for i in range(0,len(training_set)):
        train_img =cv2.imread(training_set[i])
        (kp2, des2)=orb.detectAndCompute(train_img,None)

        bf=cv2.BFMatcher()
        all_matches=bf.knnMatch(des1,des2,k=2)

        good=[]

        for(m,n) in all_matches:
            if m.distance<0.789*n.distance:
                good.append([m])

        if len(good)>max_val:
            max_val=len (good)
            max_pt=imax_kp=kp2
        print(i, ' ', training_set[i],' ',len(good))

    if max_val !=8:
        print(training_set[max_pt])
        print('good matches ',max_val)
        training_img=cv2.imread(training_set[max_pt])
        img3=cv2.drawMatchesKnn(test_img, kp1, train_img, max_kp, good, 4)
        note =str(training_set[max_pt])[6:-4]
        print('\nDetected denomination: Rs. ',note)
        talk('\nDetected denomination: Rs. ')
        talk(note)
    else:
        print('No Matches')
        talk('No Matches')
    return
talk('listening...')
while True:
    try:
        with sr.Microphone() as source:
            print('listening...')
            r.adjust_for_ambient_noise(source)
            audio=r.listen(source)
            command=r.recognize_google(audio)
            command=command.lower()
            print(command)
        if 'alexa' in command:
            words=command.replace('alexa','')
            if words=="hello":
                talk('Hello How are you')
            if words=="identify":
                talk("okay")
                name=clasify_face("test.jpg")
                for a in name:
                    talk(a)
            
            if words=="take":
                talk("okay")
                take_image(words)
            if "save" in words :
                talk("Okay")
                take_image(words)
            if "hu" in words:
                talk("okay")
                name=check_image()
                if name is None:
                    talk('not recognizable')
                else:
                    for a in name:
                        talk(a)
                if "surrounding" in words:
                    talk("okay")
                    name=check_surrounding()
                    if name is None:
                        talk('not recognizable')
                    else:
                        for a in name:
                            talk(a)
                if "read" in words:
                    talk("okay")
                    name=read()
                    talk(name)
                if "bill" in words:
                    talk("okay")
                    name=bill()
                if "currency" in words:
                    talk("okay")
                    name=currency()
                if words=="exit":
                    print("...")
                    sleep(1)
                    print("...")
                    sleep(1)
                    print("...")
                    sleep(1)
                    print("GoodBye")
                    break
    except:
        pass

