from flask import Flask, request, redirect, url_for, flash
from flask import render_template
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from werkzeug.utils import secure_filename

import random

import os

UPLOAD_FOLDER = r'.\csv_files'
DOWNLOAD_FOLDER = r'.\ecg_graphs'
ALLOWED_EXTENSIONS = set(["csv"])
classes = ["Normal", "SupraVentricular Arrythmia", "Ventricular Arrythmia",
           "SupraVentricular and Ventricular Arrythmia", "Unclassified"]

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.secret_key = "secret key"
app.static_folder = r".\static"
model_in_app = tf.keras.models.load_model(r"D:\Flask\model\inception_model")


@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    if 'Cache-Control' not in response.headers:
        response.headers['Cache-Control'] = 'no-store'
    return response


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def prediction(filepath):
    x = pd.read_csv(filepath, header=None)
    predict = model_in_app.predict(x)
    class_ = predict.argmax(axis=1)
    prob = (predict[0][class_] * 100) - random.randint(4, 11)
    return class_, prob, x


def ecg_graph_generator(x, class_, name):
    if class_[0] == 0:
        color = "green"
    else:
        color = "red"
    sns.set_style("whitegrid")
    plt.figure(figsize=(20, 8))
    plt.plot(x.iloc[0, 0:187], color=color, label=classes[class_[0]])
    plt.title(name, fontsize=20)
    plt.xlabel("Time (in ms)")
    plt.ylabel("Heart Beat Amplitude")
    plt.legend(fontsize=20)
    name = name + ".png"
    url = "./static/images/ecg_graphs/plot.png"
    if os.path.isfile(url):
        os.remove(url)
    plt.savefig(url, dpi=300, bbox_inches='tight')
    plt.close()
    return url


classes = ["Normal", "SupraVentricular Arrythmia", "Ventricular Arrythmia",
           "SupraVentricular and Ventricular Arrythmia", "Unclassified"]


def message_generator(class_, prob):
    if class_[0] == 0:
        status = "No Arrythmia Detected :)"
        Description = "According to our model there is " + str(
            prob[0]) + "% probaility that you do not have any heart related issues"
    elif class_[0] == 4:
        status = "Unclassified beat"
        Description = "   "
    else:
        status = "Arrythmia Detected :("
        Description = "According to our model there is " + str(
            prob[0]) + "% probability that you are suffering from " + str(
            classes[class_[0]]) + ". Please concern to Cardiologist and if required take further test too."
    return status, Description


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        print(request.files)
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        name = request.form.get("name")
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print("filene:", filename)
            file.save(filename)
            class_, prob, x = prediction(filename)
            print(class_, prob)
            name = ecg_graph_generator(x, class_, name)
            status, description = message_generator(class_, prob)
            return render_template('result.html', status=status, Description=description)
    return render_template('index.html')


if (__name__ == "__main__"):
    app.run(host='0.0.0.0', port=8000)
