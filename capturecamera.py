import cv2
import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
from datetime import datetime


class CaptureVideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()
    

    def get_frame(self,id,name):
        now = datetime.now() 
        
        id=str(id)
        name=str(name)
        
        b="_"
        # joining=[id,b,name]

        directory=id+b+name
        date_time = now.strftime("%H%M%S")
        date_time=int(date_time)
        success, image = self.video.read()
        small_image = cv2.resize(image, (0, 0), fx=0.75, fy=0.75)
        rgb_small_image = small_image[:, :, ::-1]
        rgb_image = rgb_small_image
        image = small_image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        parent_dir="knn_examples/train/"
        path=os.path.join(parent_dir,directory)
        if not os.path.isdir(path):
            os.mkdir(path)
        
        predictions = predict(X_img_path=gray, model_path="trained_knn_model2107.clf")
        
        cv2.imwrite(str(path)+'/'+"train%d.jpg"%date_time,image)
        for name, (top, right, bottom, left) in predictions:
            radius=right-left
            center_coordinates = (int(left+(right-left)/2), int(top+(bottom-top)/2))
        # Draw a label with a name below the face

            radius=int(radius)
            
            cv2.circle(image, center_coordinates,radius, (0, 0, 255), 2)
            
            # Display the resulting image
            break
        ret, png= cv2.imencode('.png', image)
        return png.tobytes()

        	







def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.99):
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
    return [(pred, loc) if rec else ("Unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


# def show_prediction_labels_on_image(img_path, predictions):
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")
        radius=right-left
        center_coordinates = (int(left+(right-left)/2), int(top+(bottom-top)/2))
        radius=int(radius)
        
        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.circle(center_coordinates,radius, fill=(0, 0, 255), outline=(0, 0, 255))
        
    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    pil_image.show()














    # def get_frame(self):
    #     success, image = self.video.read()
    #     image=cv2.resize(image,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
    #     gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #     face_rects=face_cascade.detectMultiScale(gray,1.3,5)
    #     for (x,y,w,h) in face_rects:
    #     	cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    #     	break
    #     ret, jpeg = cv2.imencode('.jpg', image)
    #     return jpeg.tobytes()
