import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import cv2
from pymongo import MongoClient
from datetime import datetime 

import numpy as np
import platform

#Step 1: Connect to MongoDB - Note: Change connection string as needed
client = MongoClient(port=27017)
db=client.db




# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
temp=38

# pr=['top:10','right:40','bottom:20','left:10']

def running_on_jetson_nano():
    # To make the same code work on a laptop or on a Jetson Nano, we'll detect when we are running on the Nano
    # so that we can access the camera correctly in that case.
    # On a normal Intel laptop, platform.machine() will be "x86_64" instead of "aarch64"
    return platform.machine() == "aarch64"


def get_jetson_gstreamer_source(capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=60, flip_method=0):
    """
    Return an OpenCV-compatible video source description that uses gstreamer to capture video from the camera on a Jetson Nano
    """
    return (
            f'nvarguscamerasrc ! video/x-raw(memory:NVMM), ' +
            f'width=(int){capture_width}, height=(int){capture_height}, ' +
            f'format=(string)NV12, framerate=(fraction){framerate}/1 ! ' +
            f'nvvidconv flip-method={flip_method} ! ' +
            f'video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! ' +
            'videoconvert ! video/x-raw, format=(string)BGR ! appsink'
            )






def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.9):
    # if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        # raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    # X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img_path)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img_path, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    
    # Predict classes and remove classifications that aren't within the threshold
    # return [(pred, loc,print(pred)) if rec else ("unknown", loc,print("unknown")) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
    
    return [(pred, loc) if rec else ('Unknown', loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
    
    # for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches):
    #     if rec:
    #         return (pred, loc)
    #     else:
    #         return ("unknown", loc)

# def show_prediction_labels_on_image(img_path, predictions):
    pil_image = Image.open(img_path).convert("RGB")
    
    
    draw = ImageDraw.Draw(pil_image)
    draw1 =ImageDraw.Draw(pil_image1)
    

    for name, (top, right, bottom, left) in predictions:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        
        temp=38
        name=name+" Temp="+str(temp)+"c"
            
        name = name.encode("UTF-8")
        
        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5),name, fill=(255, 255, 255, 255))
        # draw.text((left + 30, bottom - text_height - 30), temp, fill=(255, 255, 255, 255))
        # left1=50
        # bottom1=150
        # right1=150

        # #Draw a label for temperature
        # text_width1, text_height1 = draw.textsize(temp)
        # draw.rectangle(((left, bottom-text_width-10 ), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        # draw.text((left + 6, bottom - text_height - 5), temp, fill=(255, 255, 255, 255))
        
    # for temp, (top, right, bottom, left) in predictions:
    #     # Draw a box around the face using the Pillow module
    #     draw1.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
     
    #     temp = temp.encode("UTF-8")
        
    #     text_width, text_height = draw.textsize(temp)
    #     draw1.rectangle(((left, bottom - text_height - 40), (right, bottom)), fill=(0, 255, 0), outline=(0, 0, 255))
    #     draw1.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))


    # Remove the drawing library from memory as per the Pillow docs
    del draw
    del draw1

    # Display the resulting image
    pil_image.show()




prevname=[]
for q in range(0,9):
    prevname.append('Un')




def is_integer_num(n):
    if isinstance(n, int):
        return  
    if isinstance(n, float):
        return n.is_integer()
    return False


def is_integer(n):
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()


if __name__ == "__main__":

     if running_on_jetson_nano():
        # Accessing the camera with OpenCV on a Jetson Nano requires gstreamer with a custom gstreamer source string
         video_capture = cv2.VideoCapture(0)
    #video_capture = cv2.VideoCapture(get_jetson_gstreamer_source(), cv2.CAP_GSTREAMER)
     else:
        # Accessing the camera with OpenCV on a laptop just requires passing in the number of the webcam (usually 0)
        # Note: You can pass in a filename instead if you want to process a video file instead of a live camera stream
         video_capture = cv2.VideoCapture(0)



    #  video_capture = cv2.VideoCapture(0)
    #  current_time = datetime.datetime.now()
     now = datetime.now() 
     date_time = now.strftime("%d/%m/%Y ; %H:%M:%S")
     while True:
        from temp import tempe
        temp=tempe()
        
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
        rgb_small_frame = small_frame[:, :, ::-1]
        rgb_frame = rgb_small_frame
        frame = small_frame
        #rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all people in the image using a trained classifier model
        # Note: You can pass in either a classifier file name or a classifier model instance
        predictions = predict(X_img_path=rgb_frame, model_path="trained_knn_model2107.clf")
        pred=str(predictions)

        K = ','
        
        # using list comprehension + split() 
        # K Character Split String 
        res = [i for j in pred[2:].split(K) for i in (j, K)][:-1]
        name=res[0]
        # name = 'aj'
        # prevname=['Unknown','Unknown','aj','Unknown','Unknown','Unknown','Unknown','Unknown']
        # if (name in prevname): 
        #     print(name) 

        if name == '':
                name='Unknown'
                
        if (name in prevname):
            l=name
        else:
            
            
            print(name)
            removed=prevname.pop(1)
            prevname.append(name)
            # print(prevname)
            
            attendance={
                        
                        'date_time' : date_time,
                        'name' : name,
                        'temperature' : temp
                    }
            result=db.seasattendance.insert_one(attendance)
            print('printed {0}'.format(result.inserted_id))

        
  
        
            
                                     
        
      
        


        
        for name, (top, right, bottom, left) in predictions:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            

            name=name+" Temp="+str(temp)+"c"
            
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            
            # Display the resulting image
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        

       

# Release handle to the webcam

video_capture.release()
cv2.destroyAllWindows()


