from flask import Flask , render_template , url_for

from flask_pymongo import PyMongo

from bson.json_util import dumps
import os
  
from pymongo import MongoClient
from bson import ObjectId
from flask import jsonify, request

#from erkzeug.wsecurity import generate_password_hash,check_password_hash

from temp import temperature1
print(temperature1)



from datetime import datetime

app = Flask(__name__)

app.secret_key = "secret"

app.config['MONGO_URI'] = "mongodb://localhost:27017/db"  #database location
 
mongo = PyMongo(app) #databse intialised

@app.route('/')
def index():
    return render_template('index.html')




 








if __name__ == "__main__":  
    app.run(debug=True)
