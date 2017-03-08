import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.applications.vgg16 import preprocess_input, decode_predictions


def pre_processing_raw(img):# Module for preprocessing raw images, takes a single 
    img = load_img(img)                            #image as input and gives out a processd image as output 
    x = img_to_array(img)  
    x = np.expand_dims(x, axis=0)
    x=preprocess_input(x)
    return (x)

