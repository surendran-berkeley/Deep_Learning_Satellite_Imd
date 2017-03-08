import os
import numpy as np
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.applications.vgg16 import preprocess_input, decode_predictions
import img_process


## Takes in processed image data stored in Cache as data and 
## Spits out the features of the fourth conv block 

def vgg_block4(data):
    features=[]
    model = Model(input=model_dict["Vgg16"].input, output=model_dict["Vgg16"].get_layer('block4_pool').output)
    model=model.predict()
    for i in processed_data:
        preds = model(i)
        features.append(preds)
    np.save(open('vgg_block_4_fextractor.npy', 'w'), features)
    print ("....................")
    print ("Features from 4th block extracted and saved")
    print ("                     .......................")







