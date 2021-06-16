import numpy as np
from matplotlib import pyplot as plt
import os
from keras.preprocessing import image
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input
from keras.models import Model
from keras.layers import Input

directory = '../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train'

datagen = image.ImageDataGenerator(rescale = 1./255, validation_split = 0.2) 

train_set = datagen.flow_from_directory(directory, target_size = (100, 100), batch_size = 16,
                                        class_mode = 'categorical', subset = 'training')
val_set = datagen.flow_from_directory(directory, target_size = (100, 100), batch_size = 16, 
                                      class_mode = 'categorical', subset = 'validation')

#----------------------------------------------------------------------------------------------------------------#
def network(nb_class, inputsize): # nb_class = 29, input = (100, 100, 3)
    input_img = Input(shape = inputsize)
    x = Conv2D(16, (3,3), strides = (1,1), activation = 'relu', padding = 'same', name = 'gkm_conv1')(input_img) 
        # 1st - number of filters, 2nd - filter size
    x = MaxPool2D((3,3), strides = (2,2), padding = 'same', name = 'gkm_maxpool1')(x)
    x = Conv2D(32, (3,3), strides = (1,1), activation = 'relu', padding = 'same', name = 'gkm_conv2')(x)
    x = MaxPool2D((3,3), strides = (2,2), padding = 'same', name = 'gkm_maxpool2')(x)
    x = Flatten(name = 'flatten')(x)
    x = Dense(100, activation = 'relu')(x)
    x = Dense(nb_class, activation = 'softmax')(x)
    model = Model(input_img, x)
    return model
  #----------------------------------------------------------------------------------------------------------------#
  
model = network(nb_class = 29, inputsize = (100,100,3))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()

hist = model.fit(train_set, epochs = 10, validation_data = val_set, verbose = 1)


