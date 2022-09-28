from flask import Flask
from flask_restful import Resource, Api
import mysql.connector

app = Flask(__name__)

# https://flask-mysqldb.readthedocs.io/en/latest/
mydb = mysql.connector.connect(
    host="localhost",
    user="dbuser",
    password="dbpassword",
    database="spaces"
)

api = Api(app)


def writeIntoDB(temp, hum, date):
    try:
        mycursor = mydb.cursor()

        sql = "INSERT INTO arduinodata (temp, hum, date) VALUES (%s, %s, %s)"
        val = (temp, hum, date)
        mycursor.execute(sql, val)
        mydb.commit()
        return "data inserted."
    except:
        return "db writing failed"


def write_txt(temp, hum, date):
    try:
        path = "data.txt"
        update = str(temp) + "," + str(hum) + "," + date + "\n"
        my_file = open(path, "a")
        my_file.write(update)
        my_file.close()
        status = "added"
    except:
        status = "not added"
    return status


class MyData(Resource):
    def get(self, temp, hum, date):
        sta = write_txt(temp, hum, date)
        err = writeIntoDB(temp, hum, date)
        return {"job": {"temp": temp, "hum": hum, "date": date}, "statusdb": err, "statustxt": sta}


# Data should come as /date=s;hum=1,temp=1
api.add_resource(MyData, '/date=<string:date>;temp=<int:temp>;hum=<int:hum>')

if __name__ == '__main__':
    app.run(host="10.207.12.116", debug=True)
