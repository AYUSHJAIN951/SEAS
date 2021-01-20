from flask import Flask , render_template , jsonify,url_for, Response , request ,stream_with_context
from flask_pymongo import PyMongo , pymongo
from bson.json_util import dumps
import os
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
import os.path
import csv
from io import StringIO
from werkzeug.wrappers import Response


app.config['MONGO_URI'] = "mongodb://localhost:27017/db"  #database location
app.secret_key = "escret"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
mongo = PyMongo(app) #databse intialised
client = MongoClient()
db = client.db


@app.route('/download_log')
def download_log():
    
    def generate():
        
        data = StringIO()
        w = csv.writer(data)

        cursor = db.seasemployees.find({}, {'_id': 1,'name': 1,'epost':1,})
        with open('log.csv', 'w') as outfile:
           
            w.writerow(('_id','name','epost'))
            
            yield data.getvalue()
            data.seek(0)
            data.truncate(0)
            for answers_record in cursor:  # Here we are using 'cursor' as an iterator
                w.writerow((
                    answers_record['_id'],
                    answers_record['name'],
                    answers_record['epost']
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


# if __name__ == "__main__":
#     app.run(debug=True)



# Sources:-
# https://stackoverflow.com/questions/28011341/create-and-download-a-csv-file-from-a-flask-view
# https://stackoverflow.com/questions/40245873/export-data-to-csv-from-mongodb-by-using-python