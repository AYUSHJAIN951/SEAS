from flask import Flask , render_template , url_for, Response , request
from flask_pymongo import PyMongo , pymongo
from bson.json_util import dumps
import os
from pymongo import MongoClient
from bson import ObjectId
from flask import jsonify, request
from temp import tempe
import cv2
import math
from datetime import datetime
import os.path

import csv
from datetime import datetime
from io import StringIO
from flask import Flask

from flask import stream_with_context
from werkzeug.wrappers import Response

app = Flask(__name__)
app.secret_key = "secret"
app.config['MONGO_URI'] = "mongodb://localhost:27017/db"  #database location
app.secret_key = "escret"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
mongo = PyMongo(app) #databse intialised
client = MongoClient()
db = client.db

# _id:5f219463628ccfb1ae7c252d
# Empid:"E1001"
# date_time:"29/07/2020 ; 20:53:14"
# name:"Vivek"
# temperature:35.65
# uv:"yes




# Step 1: Create the list of dictionaries (one dictionary per entry in the `answers` list)
# cursor = db.seasattendance2.find(
#     {}, {'_id': 1, 'Empid': 1, 'date_time': 1, 'name': 1,'temperature': 1,'uv': 1})

# with open('stack_039.csv', 'w') as outfile:
#     fields = ['_id', 'Empid', 'date_time','name','temperature','uv']
#     write = csv.DictWriter(outfile, fieldnames=fields)
#     write.writeheader()
#     for answers_record in cursor:  
#         # Here we are using 'cursor' as an iterator
#         answers_record_id = answers_record['_id']
#         for answer_record in answers_record['answers']:
#             flattened_record = {
#                 '_id': answers_record_id,
#                 'Empid': answer_record['Empid'],
#                 'date_time': answer_record['date_time'],
#                 'name': answer_record['name'],
#                 'temperature': answer_record['temperature'],
#                 'uv': answer_record['uv']
#             }
#             write.writerow(flattened_record)


# app = Flask(__name__)

# example data, this could come from wherever you are storing logs
# log = [
#     ('login', datetime(2015, 1, 10, 5, 30)),
#     ('deposit', datetime(2015, 1, 10, 5, 35)),
#     ('order', datetime(2015, 1, 10, 5, 50)),
#     ('withdraw', datetime(2015, 1, 10, 6, 10)),
#     ('logout', datetime(2015, 1, 10, 6, 15))
# ]

@app.route('/')
def download_log():
    
    def generate():
        
        data = StringIO()
        w = csv.writer(data)

        cursor = db.seasatt.find({}, {'_id': 1,'name': 1,'epost':1})
        with open('stack_039.csv', 'w') as outfile:
            # fields = ['_id','name','epost']
            w.writerow(('_id','name','epost'))
            # write = csv.DictWriter(outfile, fieldnames=fields)
            # write.writeheader()
            yield data.getvalue()
            data.seek(0)
            data.truncate(0)
            for answers_record in cursor:  # Here we are using 'cursor' as an iterator
                w.writerow((
                    answers_record['_id'],
                    answers_record['name'],
                    answers_record['epost']
                )) 
                # flattened_record = {
                #         '_id': answers_record['_id'],
                #         'name': answers_record['name'],
                #         'epost': answers_record['epost']
                #         }
                yield data.getvalue()
                data.seek(0)
                data.truncate(0)

            #     write.writerow(flattened_record)
            # for item in log:
    #         w.writerow((
    #             item[0],
    #             item[1].isoformat()  # format datetime as string
    #         ))
    #         yield data.getvalue()
    #         data.seek(0)
    #         data.truncate(0)



    #     data = StringIO()
    #     w = csv.writer(data)

    #     # write header
    #     w.writerow(('action', 'timestamp'))
    #     yield data.getvalue()
    #     data.seek(0)
    #     data.truncate(0)

    #     # write each log item
    #     for item in log:
    #         w.writerow((
    #             item[0],
    #             item[1].isoformat()  # format datetime as string
    #         ))
    #         yield data.getvalue()
    #         data.seek(0)
    #         data.truncate(0)

    # stream the response as the data is generated
    response = Response(generate(), mimetype='text/csv')
    # add a filename
    response.headers.set("Content-Disposition", "attachment", filename="stack_039.csv")
    return response


@stream_with_context
def generate():
    ...


if __name__ == "__main__":
    app.run(debug=True)

