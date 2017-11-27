import os
from flask import Flask, render_template, request, flash, redirect, url_for
from flask_uploads import UploadSet, configure_uploads, IMAGES
from imageio import imread
from skimage.transform import resize
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

import sys 
import os
sys.path.append(os.path.abspath("./models"))
from load import * 

app = Flask(__name__)
app.secret_key = 'random string'

photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = '/home/gopi34/Desktop/mini_project/static/uploads'
configure_uploads(app, photos)

global model, graph
model, graph = init()

@app.route("/")
def hello():
    return render_template('index.html') 

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST' and 'photo' in request.files:
        image_file = request.files['photo']
        image_file = imread(image_file)
        resized_image = resize(image_file, (150, 150))
        reshaped_image = resized_image.reshape(1, resized_image.shape[0], resized_image.shape[1], resized_image.shape[2])
        print(reshaped_image.shape)
        with graph.as_default():
            pred = model.predict(reshaped_image)
            print(pred)
            if (pred <=0.49):
                pred = 'Cat'
                print('It\'s a cat. \n')
                #flash("My intuition tells that the image is .......\n")
                #flash(pred)
            elif(pred  >= 0.55):
                pred = 'Dog'
                print('It\'s a dog. \n')
            else:
                pred = 'Aham bhramasmi !!!'
                print('Aham bhramasmi !!!')
            flash(pred)
            #return redirect(url_for(output))
        #return pred
        return redirect(url_for('hello'))
    #return render_template("index.html", result=pred)

@app.route("/feedback")
def feedback():
    return render_template('feedback.html') 

@app.route("/contact")
def contact():
    return render_template('contact.html') 

if __name__ == "__main__":
    app.run(debug = True)
    app.run()
