import os.path
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.applications.vgg16 import preprocess_input, decode_predictions

def block_4():
    if os.path.isfile(vgg16_weights.h5)==False:
    	print " The weights fie is not available"
    	print "Save the file, breaking off for now"
    	return (None)
    	
    vgg_weights="vgg16_weights.h5" # Provide the weights location path 
    load_weights=hd5py.file(vgg_weights)
    
    conv_model= sequential()
    conv_model.add(ZeroPadding2D((1, 1)))
    conv_model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    conv_model.add(ZeroPadding2D((1, 1)))
    conv_model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    conv_model.add(ZeroPadding2D((1, 1)))
    conv_model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    conv_model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    ## Load weights from 24-30 conv layer and assign it to the conv_model (5th conv block )
        ## as initial weights 
    for i in range(f.attrs["nb_layer"]):
        if i >=24 and i<=30:
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            model.layers[k].set_weights(weights)
        else:
            pass
    
    f.close()
    
    #top_model_weights_path = "block 2 output"
    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(3, activation='sigmoid'))

    # note that it is necessary to start with a fully-trained
    # classifier, including the top classifier,
    # in order to successfully do fine-tuning
    
    ### Commention out the top_model_load_weights to check how random weights may play 
    #top_model.load_weights(top_model_weights_path)
    
    conv_model.add(top_model)
    
    conv_model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])
    
    conv_model.fit(train_data, train_labels,
          nb_epoch=50, batch_size=32,
          validation_data=(validation_data, validation_labels))
    