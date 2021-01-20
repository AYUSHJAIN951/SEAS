from flask import Flask , render_template ,stream_with_context, url_for, Response , request,jsonify,redirect
from flask_pymongo import PyMongo , pymongo
from flask_bootstrap import Bootstrap
from bson.json_util import dumps
import os
from pymongo import MongoClient
from bson import ObjectId
from temp import tempe
import cv2
import math
from sklearn import neighbors
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
from camera import VideoCamera
from cameracopy import VideoCamera2
from datetime import datetime
from capturecamera import CaptureVideoCamera #for saving new user pictures
import codecs
from werkzeug.utils import secure_filename
import os.path
import csv
from io import StringIO
from werkzeug.wrappers import Response
import shutil
import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
from face_train import train
# from app import train
UPLOAD_FOLDER = '/static/employeeimages'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
Bootstrap(app)

app.secret_key = "secret"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MONGO_URI'] = "mongodb://localhost:27017/db"  #database location
app.secret_key = "escret"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
mongo = PyMongo(app) #databse intialised

# FOR CONNECTING TO MONGO ONLINE
# app.config['MONGO_DBNAME'] = 'database_name'
# app.config['MONGO_URI'] = 'mongodb://db_name:db_password@ds123619.mlab.com:23619/db_table_name'

client = MongoClient()
mongoii = os.getenv('MONGODB')
clientii = MongoClient(mongoii)
db = client.db
todos = db.seasemployees #Select the collection name
dbs = client.db    
atodos = dbs.seasattendance
m=0
n=[]    
nn=[]
access = 0




def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
flag=0
@app.route('/', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] == 'admin' or request.form['password'] == 'admin':
            flag=1
            return redirect('/index')
        elif  request.form['username'] == 'normal' or request.form['password'] == 'normal':
            flag=2
            return redirect('/index')
        else:
            error = 'Invalid Credentials. Please try again.'
            
    return render_template('login.html', error=error)

@app.route('/index')
def index():
    return render_template('index.html')

def gene(camera):
    while True:
        frame = camera.get_frame()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n'  )
def gene2(cameracopy):
    while True:
        
        frame2 = cameracopy.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n\r\n'  )       
                     

@app.route('/video_feed')
def video_feed():
    return Response(gene(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/video_feed3')
def video_feed3():
    return Response(gene2(VideoCamera2()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/profile/<_id>')
def profile(_id): 
    user = mongo.db.seasemployees.find_one_or_404({'_id' : _id})
    return render_template('profileview.html' , user=user,_id=_id)


@app.route('/Train') 
def train():

    from face_train import train
    
    
    classifier = train("knn_examples/train", model_save_path="trained_knn_model.clf", n_neighbors=11)
    

    return render_template('Train.html')


@app.route("/addnew2", methods=["GET","POST"])    

def addnew2():    
    #Adding a Task  
    if 'empimage' in request.files:
        id=request.values.get("id") 
        name=request.values.get("name")
        empimage=request.files['empimage']
        mongo.save_file(empimage.filename,name)

        # empimage.save(os.path.join('/static/employeeimages',name,".png"))
        # id = grid_fs.put(empimage, content_type = pimage.content_type, filename = name)
        epost=request.values.get("epost")    
        bloodgroup=request.values.get("bloodgroup")    
        mobile_no=request.values.get("mobile_no")    
        db.seasemployees.insert({ '_id': id , "name": name, "epost": epost ,'empimage' : empimage.filename, "bloodgroup": bloodgroup, "mobile_no":mobile_no})    
        
    return render_template('addnew2.html') 
      
@app.route("/newempcreated") 



@app.route('/file/<filename>')
def file(filename):
    return  mongo.send_file(filename)

@app.route("/list")    
def lists ():
        
    #Display the all Tasks    
    todos_l = todos.find()    
    a1="active"
    name=todos["name"]
    return render_template('list.html',a1=a1,todos=todos_l)    



@app.route("/attendance")    
def attendance ():    
    #Display the Attendance    
    atodos_l = atodos.find()   
    a1="active"    
    name=atodos["name"]
    return render_template('attendance.html',a1=a1,atodos=atodos_l)    
    
@app.route("/checkbyid")    
def checkbyid ():    
    #Display the Attendance    
    atodos_l = atodos.find()   
    a1="active"    
    return render_template('checkbyid.html',a1=a1,atodos=atodos_l)    
    



@app.route("/action", methods=['POST'])    
def action ():    
    #Adding a New Employee
    if 'empimage' in request.files:     
        id=request.values.get("id")
        name=request.values.get("name")
        empimage=request.files['empimage']
        mongo.save_file(empimage.filename,name)    
        epost=request.values.get("epost")    
        bloodgroup=request.values.get("bloodgroup")    
        mobile_no=request.values.get("mobile_no")  
        db.seasemployees.insert({"_id":id}, {'$set':{"name": name, "epost": epost ,'empimage' : empimage.filename,  "bloodgroup": bloodgroup, "mobile_no":mobile_no }})     
    return redirect('/index')
@app.route("/index")
@app.route("/action3", methods=['POST'])    
def action3(): 

    #Updating a Task with various references    
    id=request.values.get("_id")
    name=request.values.get("name")
    # empimage=request.files['empimage']
    # mongo.save_file(empimage.filename,name)    
    epost=request.values.get("epost")    
    bloodgroup=request.values.get("bloodgroup")    
    mobile_no=request.values.get("mobile_no")  
    db.seasemployees.update({"_id":id}, {'$set':{"name": name, "epost": epost ,"bloodgroup": bloodgroup, "mobile_no":mobile_no }})     
    return redirect('/index')
directory =""
# for Adding new member
@app.route("/action2", methods=['POST'])    
def action2 ():

    #Adding a Task
    if 'empimage' in request.files:
         
        id=request.values.get("id")    
        name=request.values.get("name")
        empimage=request.files["empimage"]
        mongo.save_file(name,empimage)     
        # empimage.save(os.path.join('/static/employeeimages',name,".png"))
        
        epost=request.values.get("epost")    
        bloodgroup=request.values.get("bloodgroup")    
        mobile_no=request.values.get("mobile_no")  
       
        insert={"_id": id ,
                "name": name,
                "epost": epost ,
                "empimage" : empimage.filename, 
                "bloodgroup": bloodgroup,
                "mobile_no":mobile_no }
        db.seasemployees.insert_one(insert)
        
        
         
    return render_template('detailverifyaddnew2.html',id=id,epost=epost,name=name)
     



id=""
name=""
@app.route('/camerapicstrain/<id>/<name>', methods=['POST']) 
   
def capturepicstrain(id,name):
    if request.method == 'POST':
        id=request.values.get("id") 
        name=request.values.get("name")
        epost=request.values.get("epost")
       
    return render_template('captureindex.html', directory=directory , id=id,name=name)

@app.route('/capturecamera', methods=['POST'])
def gen(capturecamera,id,name):
    while True:
        
        frame = capturecamera.get_frame(id,name)
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    


@app.route('/video_feed2/<id>/<name>')
def video_feed2(id,name):
    return Response(gen(CaptureVideoCamera(),id,name),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/capturepicstrain", methods=["POST"])

@app.route("/buttontoindex" , methods=['POST'])
def buttontoindex():
    return redirect('/index')

@app.route("/update")    
def update():    
    id=request.values.get("_id")    
    task=todos.find({"_id":id})    
    return render_template('update.html',tasks=task)    
  
@app.route("/remove")    
def remove ():    
    #Deleting a Task with various references    
    key=request.values.get("_id")
    res=[]
    listofdirectories=[]
    directory=key
    
# Parent Directory path 
    parent_dir="knn_examples/train/"
    path=os.path.join(parent_dir)
    listofdirectories=[f.name for f in os.scandir(path) if f.is_dir()]
    res = list(filter(lambda x: key in x, listofdirectories))
    
    path=os.path.join(parent_dir,res[0])   
    print(path)
    print(res[0])
    print(listofdirectories)
    if os.path.isdir(path):
        shutil.rmtree(path)
        # os.rmdir(path)
        
            
        todos.remove({"_id":key}) 
    return redirect('/index')

@app.route("/remove2")    
def remove2 ():    
    #Deleting a Task with various references    
    key=request.values.get("_id")    
    atodos.remove({"_id":ObjectId(key)})    
    return redirect('/index')
@app.route("/search", methods=['GET'])    
def search():    
    #Searching a Task with various references    
    
    key=request.values.get("key")    
    refer=request.values.get("refer")    
    if(key=="_id"):    
        atodos_l = atodos.find({refer:ObjectId(key)})    
    else:    
        atodos_l = atodos.find({refer:key})    
    return render_template('checkbyid.html',atodos=atodos_l,key=key,refer=refer)    
    




@app.route('/download_log')
def download_log():
    # if request.method == 'POST':
    #     nameeee=request.values.get("nametodownload") 
    def generate():
        
        data = StringIO()
        w = csv.writer(data)

        cursor = db.seasemployees.find({}, {'_id': 1,'name': 1,'epost':1,'bloodgroup':1,'mobile_no':1})
        with open('log.csv', 'w') as outfile:
           
            w.writerow(('_id','name','epost','bloodgroup','mobile_no'))
            
            yield data.getvalue()
            data.seek(0)
            data.truncate(0)
            for answers_record in cursor:  # Here we are using 'cursor' as an iterator
                w.writerow((
                    answers_record['_id'],
                    answers_record['name'],
                    answers_record['epost'],
                    answers_record['bloodgroup'],
                    answers_record['mobile_no']
                )) 
               
                yield data.getvalue()
                data.seek(0)    
                data.truncate(0)

    # stream the response as the data is generated
    response = Response(generate(), mimetype='text/csv')
    # add a filename
    response.headers.set("Content-Disposition", "attachment", filename="log.csv")
    return response





@stream_with_context
def generate():
    ...


@app.route('/download_log_attendance')
def download_log_attendance():
    # if request.method == 'POST':
    #     nameeee=request.values.get("nametodownload") 
    def generate():
        
        data = StringIO()
        w = csv.writer(data)
        cursor = db.seasattendance.find({},{'_id':1,'Empid':1,'date_time':1,'name':1,'temperature':1,'uv':1})
        # cursor = db.seasattendance.find({}, {'_id':1,'date_time': 1,'Empid': 1,'name':1,'temperature':1})
        with open('log.csv', 'w') as outfile:
           
            w.writerow(('date_time','Empid','name','temperature'))
            
            yield data.getvalue()
            data.seek(0)
            data.truncate(0)
            for answers_record in cursor:  # Here we are using 'cursor' as an iterator
                w.writerow((
                    answers_record['date_time'],
                    answers_record['Empid'],
                    answers_record['name'],
                    answers_record['temperature'],
                    
                )) 
               
                yield data.getvalue()
                data.seek(0)    
                data.truncate(0)

    # stream the response as the data is generated
    response = Response(generate(), mimetype='text/csv')
    # add a filename
    response.headers.set("Content-Disposition", "attachment", filename="attendance_log.csv")
    return response





@stream_with_context
def generate():
    ...



@app.route('/download_log_attendance_search/<refer>/<key>')
def download_log_attendance_search(refer,key):
    # if request.method == 'POST':
    #     nameeee=request.values.get("nametodownload") 
    def generate(refer,key):
        keyy = refer
        key=key
        data = StringIO()
        w = csv.writer(data)
        cursor = db.seasattendance.find({keyy:key},{'_id':1,'Empid':1,'date_time':1,'name':1,'temperature':1,'uv':1})
        # cursor = db.seasattendance.find({}, {'_id':1,'date_time': 1,'Empid': 1,'name':1,'temperature':1})
        with open('log.csv', 'w') as outfile:
           
            w.writerow(('date_time','Empid','name','temperature'))
            
            yield data.getvalue()
            data.seek(0)
            data.truncate(0)
            for answers_record in cursor:  # Here we are using 'cursor' as an iterator
                w.writerow((
                    answers_record['date_time'],
                    answers_record['Empid'],
                    answers_record['name'],
                    answers_record['temperature'],
                    
                )) 
               
                yield data.getvalue()
                data.seek(0)    
                data.truncate(0)

    # stream the response as the data is generated
    response = Response(generate(refer,key), mimetype='text/csv')
    # add a filename
    response.headers.set("Content-Disposition", "attachment", filename="attendance_log.csv")
    return response





@stream_with_context
def generate():
    ...


if __name__ == "__main__":
    app.run(debug=True)

