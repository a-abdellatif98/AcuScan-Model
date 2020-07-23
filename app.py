import os
from flask import Flask, request, jsonify
from numpy import loadtxt
from werkzeug.utils import secure_filename

import numpy as np

import keras 
import tensorflow.keras
from keras.models import load_model
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from tensorflow.keras.losses import categorical_crossentropy
from PIL import Image


UPLOAD_FOLDER = './update'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return 'hello, world!'


@app.route('/analyze_image', methods=['GET', 'POST'])
def analyze_image():
    if 'file' not in request.files:
        return 'no files sent!'

    file = request.files['file']
    if file.filename == '':
        return 'no files sent!'

    if file and allowed_file(file.filename):
        classes = ['Cardiomegaly', 'Emphysema','Effusion', 'No Finding', 'Hernia', 'Infiltration', 'Mass', 'Nodule','Pneumothorax', 'Pleural_Thickening', 'Fibrosis', 'Atelectasis','Pneumonia']
        filename = secure_filename(file.filename)

        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        vgg16 = VGG16(weights=None , include_top=True)
        model = keras.models.Sequential()
        for layer in vgg16.layers[0:19]:
            model.add(layer)
        model.add(keras.layers.Conv2D(16,(1,1),input_shape=(512,512,7)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(4, (1,1), strides=(1, 1), padding='valid',activation="relu"))
        model.add(keras.layers.AvgPool2D(pool_size=2, strides=2))
        model.add(keras.layers.GlobalMaxPooling2D())
        model.add(Dense(13, activation='sigmoid'))
        model.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
        model.load_weights("PreTraindModel.h5")


        img = image.load_img("./update/"+filename,target_size=(224,224,3))
        img = image.img_to_array(img)
        img = img/255
        # run the inference
        proba = model.predict(img.reshape(1,224,224,3))
        top_3 = np.argsort(proba[0])[:-4:-1]

    
        result = jsonify( {'first': classes[top_3[0]], 'second': classes[top_3[1]],'third': classes[top_3[2]] } )
        return result


