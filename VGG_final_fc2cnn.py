import os.path
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.applications.vgg16 import preprocess_input, decode_predictions

def block_4(data, labels, predict=False):

    if os.path.isfile(vgg16_weights.h5)==False:
    	print " The weights fie is not available"
    	print "Save the file, breaking off for now"
    	return (None)
    

    train_data=data ### ??????
    train_labels=labels #### ??????
    validation_data = #??????
    validation_labels = #### ?????


    input_shape=data[0].shape	
    vgg_weights="vgg16_weights.h5" # Provide the weights location path 
    load_weights=hd5py.file(vgg_weights)
    

    #### Conv_Block_5

    conv_model= sequential()
    conv_model.add(ZeroPadding2D((1, 1), input_shape=(input_shape)))
    conv_model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    conv_model.add(ZeroPadding2D((1, 1)))
    conv_model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    conv_model.add(ZeroPadding2D((1, 1)))
    conv_model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    conv_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    ### Conv_Block_6
    ### Convert the fully connected part to conv layer 
    conv_model.add(Convolution2D(4096, 7, 7), activation='relu')
    conv_model.add(Convolution2D(4096, 1,1), activation='relu')
    conv_model.add(Convolution2D(1000, 1,1), activation='relu')







    ## Load weights from 24-30 conv layer and assign it to the conv_model (5th conv block )
        ## as initial weights 
    for i in range(f.attrs["nb_layer"]):
        if i >=24 and i not in [31,33,,35]: ## Note; Model weights are indexed from zero 
        ## Also note: 31, 33,35 are droput and flatten so skipping them 
            g = f['layer_{}'.format(i)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            model.layers[i].set_weights(weights)
        else:
            pass
    
    f.close()
    
    #top_model_weights_path = "block 2 output"
    top_model = Sequential()
    top_model.add(Flatten())
    top_model.add(Dense(256, activation='relu')) ## 256 is arbitray 
    top_model.add(Dropout(0.5))
    top_model.add(Dense(3, activation='softmax'))


    
    ### Commention out the top_model_load_weights to check how random weights may play 
    #top_model.load_weights(top_model_weights_path)
    
    conv_model.add(top_model)
    
    conv_model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])
    
    conv_model.fit(train_data, train_labels,
          nb_epoch=50, batch_size=32,
          validation_data=(validation_data, validation_labels))


    model_weights=conv_model.get_weights()

    ### Write numpy code to save weights and load them back
    