from flask import Flask, render_template, request, send_from_directory, flash, url_for
import cv2
import urllib.request
from main import getPrediction
from cv2 import cv2
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten
import numpy as np 
import pickle, os, sys
from werkzeug.utils import secure_filename

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
model.load_weights('final_model_covid_detection.hdf5')
model.save('filename')

#filename='finalized_model.sav'
#pickle.dump(model,open("model_pickle", 'wb'))
#loaded_model=pickle.load(open("model_pickle", 'rb'))

COUNT = 0
#sys.path.append(os.path.abspath('./model'))
app = Flask(__name__, template_folder='templates')
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route('/', methods=['GET'])
def man():
    return render_template('index.jinja2')

@app.route('/', methods=['POST'])
def submit_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            getPrediction(filename)
            label, acc = getPrediction(filename)
            flash(label)
            flash(acc)
            flash(filename)
            return redirect('/')

#@app.route('/home', methods=['GET','POST'])
#def home():
 #   global COUNT
  #  if request.method=='POST':
   #     img = request.files['image']

    #img.save('SDS_1/{}.jpg'.format(COUNT))    
        #img_arr = cv2.imread(img)
        #img.save(secure_filename(img.filename))
        #cv2.imwrite('img', img_arr)
        #COUNT += 1
        #return 'image uploaded'
        #return render_template('prediction.jinja2')
    
    #img_arr = cv2.resize(img_arr, (244,244))
    #img_arr = img_arr / 255.0
    #img_arr = img_arr.reshape(1, 244,244,3)
    #prediction = model.predict(img_arr)

    
    #return render_template('prediction.jinja2', data=prediction)

#@app.route('/load_img')
#def load_img():
 #   global COUNT
  #  return send_from_directory('SDS_1', "{}.jpg".format(COUNT-1))

if __name__ == '__main__':
    #port = int(os.environ.get('PORT', 5000))
    app.run()