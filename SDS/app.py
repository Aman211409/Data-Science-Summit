from __future__ import division, print_function
import keras
from keras.models import *
from keras.layers import *
import cv2
from keras.utils.vis_utils import plot_model
from keras.callbacks import  EarlyStopping
import pickle
import sys, os, gc, glob, re
import numpy as np
import pandas as pd
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

#app = Flask(__name__)

#model_path = "C:/Users/LENOVO/Desktop/SDS/final_model_covid_detection.hdf5"
model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = (224, 224, 3)))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience = 3)
model.load_weights("final_model_covid_detection.hdf5")
#model._make_predict_function()
#pickle.dump(model, open('model.pkl', "wb"))
#model = pickle.load(open('model.pkl', "rb"))
#We built a complete function for image processing

def predict(img_path, model):
    img = image.load_image(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.true_divide(x, 255)
    #x = preprocess_input(x)
    preds = model.predict(x)
    return preds
    #test_preprocess = keras.preprocessing.image.ImageDataGenerator(
     #                                       rescale=1./255)

    #test = test_preprocess.flow_from_directory(
     #                                       test_path_variable,
      #                                      target_size = (224, 224),
       #                                     batch_size = 16,
        #                                    class_mode = 'binary'
         #                                   )


COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1


#@app.route('/', methods=['GET'])
#def index():
 #   return render_template('../index.html')

#@app.route('/predict', methods=['GET', 'POST'])
#def upload():
 #   if request.method == 'POST':

#        f = request.files['file']

 #       basepath = os.path.dirname(__file__)
  #      file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))

   #     f.save(file_path)

    #    preds = predict(file_path, model)

     #   pred_class = decode_predictions(preds, top=1)
      #  result = str(pred_class[0][0][1])
       # return result
    #return None
@app.route('/', methods=['GET'])
def man():
    return render_template('index_1.html')


@app.route('/home', methods=['POST'])
def home():
    global COUNT
    img = request.files['image']

    img.save('final_model_covid_detection/{}.jpg'.format(COUNT))    
    img_arr = cv2.imread('final_model_covid_detection/{}.jpg'.format(COUNT))

    img_arr = cv2.resize(img_arr, (244,244))
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 244,244,3)
    prediction = model.predict(img_arr)


if __name__ == '__main__':
    app.run(debug=True)