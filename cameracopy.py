import cv2
import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
from pymongo import MongoClient
from datetime import datetime 
import numpy as np
import platform
import face_recognition
import cv2
from datetime import datetime, timedelta
import numpy as np
import platform
import pickle
from pymongo import MongoClient
from datetime import datetime 


#Step 1: Connect to MongoDB - Note: Change connection string as needed
client = MongoClient(port=27017)
db=client.db

# Our list of known face encodings and a matching list of metadata about each face.
known_face_encodings = []
known_face_metadata = []



nameverify=['U','W', 'T','W', 'T']
prevname=[]
for q in range(0,4):
    prevname.append('Un')




class VideoCamera2(object):
    def __init__(self):

        if running_on_jetson_nano():
        # Accessing the camera with OpenCV on a Jetson Nano requires gstreamer with a custom gstreamer source string
            # self.video = cv2.VideoCapture(0)
            # self.video = cv2.VideoCapture("http://192.168.1.112/videostream.cgi?user=admin&pwd=123456&resolution=32")

            self.video = cv2.VideoCapture("https://192.168.1.4:4343/video")
    #video_capture = cv2.VideoCapture(get_jetson_gstreamer_source(), cv2.CAP_GSTREAMER)
        else:
        # Accessing the camera with OpenCV on a laptop just requires passing in the number of the webcam (usually 0)
        # Note: You can pass in a filename instead if you want to process a video file instead of a live camera stream
            self.video = cv2.VideoCapture("https://192.168.1.4:4343/video")
            # self.video = cv2.VideoCapture(0)
            # self.video = cv2.VideoCapture("http://192.168.1.112/videostream.cgi?chn=1&user=admin&pwd=123456&resolution=32")
        # self.video = cv2.VideoCapture(rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp)
        # 0 for web camera live stream      
#       for cctv camera'rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp'
#  example of cctv or rtsp: 'rtsp://mamun:123456@101.134.16.117:554/user=mamun_password=123456_channel=1_stream=0.sdp'
    def __del__(self):
        self.video.release()

    
    def get_frame(self):
        

        splitting=[]
        namesplit=[]
        now = datetime.now() 
        date_time = now.strftime("%d/%m/%Y ; %H:%M:%S")
        date_time2 = now.strftime("%d%m%Y%H%M%S")
        date_time2=int(date_time2)
        from temp import tempe
        temp=tempe()
        from uv import uv
        uv=uv()
        success, image = self.video.read()  
        small_image = cv2.resize(image, (0, 0), fx=0.75, fy=0.75)
        rgb_small_image = small_image[:, :, ::-1]
        rgb_image = rgb_small_image
        image = small_image

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        nofacedetected=0
        predictions = predict(X_img_path=gray, model_path="trained_knn_model.clf")
        # print(predictions)
        if predictions==[]:
            nofacedetected=1
        pred=str(predictions)

        K = ','
        door="close"
        # using list comprehension + split() 
        # K Character Split String 
        res = [i for j in pred[3:].split(K) for i in (j, K)][:-1]
        # print(res)
        name=res[0]
        name=name[:-1]
        nametoprint=name
        
        if name == '':
            name="Unknown"
        removedname=nameverify.pop(0)
        nameverify.append(name)
        
        flag2=0
        if str(nameverify[3])==str(nameverify[4]) and str(nameverify[2])==str(nameverify[1]) and str(nameverify[3])==str(nameverify[2]) and str(nameverify[1])==str(nameverify[0])  :
            flag=1
            flag2=1
            name=nameverify[4] 
            # print(nameverify)
        else:
            flag=0

       

        namee="Unknown"
        Empid="NA"
        if (name!="Unknown"):
            namesplit=name.split('_')
            namee=namesplit[1]
            Empid=namesplit[0]
        
        if flag==1:    
            flag=0       
            if (name in prevname):
                pp=0
            else:
                if namee=="Unknown":
                    directory="Unknown Faces"
                    parent_dir="knn_examples/To_be_verified/"
                    path=os.path.join(parent_dir,directory)  
                    if not os.path.isdir(path):
                        os.mkdir(path)
                    cv2.imwrite(str(path)+'/'+"%d.jpg"%date_time2,image)
                    if nofacedetected==0:    
                        cv2.imwrite(str(path)+'/'+"%d.jpg"%date_time2,image)
                    l=name
                    
                    # result=db.seasattendance2.insert_one(attendance)
                    # print('printed {0}'.format(result.inserted_id))
                    
                else:
                    removed=prevname.pop(1)
                    prevname.append(name)
                    # print(prevname)
                    directory=str(name)
                    #TO SAVE TO A TEMPERORY FOLDER FOR VERIFICATION
                    parent_dir="knn_examples/To_be_verified/"
                    path=os.path.join(parent_dir,directory)  
                    if not os.path.isdir(path):
                        os.mkdir(path)
                  
                    # cv2.imwrite(str(path)+'/'+"%d.jpg"%date_time2,image)
                    print(namee)       
                    attendance={ 
                                'Empid' : Empid,
                                'date_time' : date_time,
                                'name' : namee,
                                'temperature' : temp,
                                'uv' : uv
                            }
                    door="open"
                    result=db.seasattendance.insert_one(attendance)
                    print('printed {0}'.format(result.inserted_id))
                    flag=0
                    cv2.imwrite(str(path)+'/'+"%d.jpg"%date_time2,image)
                    


        user_input=door
        from arduinodoorunlock import door_on_off
        door_on_off(user_input)

            
        for name, (top, right, bottom, left) in predictions:
            cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
            

            name=nametoprint+" Temp="+str(temp)+"c"
            if(flag2==1):
                cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), 2)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText( image, name, (left + 6, bottom - 6),font, 1.0, (255, 255, 255), 1)
                flag2=0
            # Display the resulting image
            break
        ret, png= cv2.imencode('.png', image)
        return png.tobytes()

        video.release()
        cv2.destroyAllWindows()

        	







def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.4):
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
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=11)
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
       
    # Remove the drawing library from memory as per the Pillow docs
    del draw
    del draw1
    pil_image.show()

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