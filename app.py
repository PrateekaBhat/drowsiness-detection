import os

from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        if request.form.get('action') == 'Face Authentication':
            os.system("python face_recognizer.py")
        else:
            pass  # unknown
    elif request.method == 'GET':
        return render_template('index.html')

    return render_template("index.html")


if __name__ == '__main__':
   app.run()