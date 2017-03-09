import os
import numpy as np
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16

## Takes in processed image data stored in Cache as data and
## Spits out the features of the fourth conv block

def vgg_block4(data):
    features=[]
    model_vgg = VGG16(weights='imagenet', include_top=False)
    model = Model(input=model_vgg.input, output=model_vgg.get_layer('block4_pool').output)
    #model = Model(input=model_dict["Vgg16"].input, output=model_dict["Vgg16"].get_layer('block4_pool').output)
    predictions = model.predict
    for i in data:
        prediction = predictions(i)
        features.append(prediction)
    np.save(open('vgg_block_4_fextractor.npy', 'wb'), features)
    print ("....................")
    print ("Features from 4th block extracted and saved")
    print ("                     .......................")
