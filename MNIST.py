#================================================================================================================#
# Training and Validation
#================================================================================================================#

# Libraries needed:

# - numpy (array)
# - matplotlib (plotting)
# - keras (deep learning) - image, convolution, maxpool, flatten, dense, models
# - os (directory reading)

# `image` = read images  
# `model` = compile everything as a model so next time can use  
# `flow_from_directory` = read image >> convert to array >> concatenate all images into big array

#----------------------------------------------------------------------------------------------------------------#

import numpy as np
from matplotlib import pyplot as plt
import os
from keras.preprocessing import image
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.models import Model
from keras.layers import Input # function of determining the input size in the first layer

directory = '../input/mnistasjpg/trainingSet/trainingSet/'

# image reader
datagen = image.ImageDataGenerator(rescale = 1./255, validation_split = 0.2) 
train_set = datagen.flow_from_directory(directory, target_size = (100, 100), batch_size = 16,
                                        class_mode = 'categorical', subset = 'training')
val_set = datagen.flow_from_directory(directory, target_size = (100, 100), batch_size = 16, 
                                      class_mode = 'categorical', subset = 'validation')

#----------------------------------------------------------------------------------------------------------------#

def network(nb_class, inputsize): # nb_class = 10, input = (100, 100, 3)
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

model = network(nb_class = 10, inputsize = (100,100,3))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()

# Save the model
model.save('SimpleDL.h5')

#----------------------------------------------------------------------------------------------------------------#

# training and validation
hist = model.fit(train_set, epochs = 20, steps_per_epoch = 25, validation_data = val_set, verbose = 1)

# plot training loss and validation loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.legend(['training loss', 'validation loss'], loc = 'upper right')

# plot training accuracy and validation accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.legend(['accuracy', 'validation accuracy'], loc = 'lower right')


#================================================================================================================#
# Testing
#================================================================================================================#

import cv2  # open cv (read images)

# read the picture to be tested
direct = '../input/mnistasjpg/testSet/testSet/img_10.jpg'
img = cv2.imread(direct)
img = cv2.resize(img, (100, 100))
data = np.array(img)/255

# DL trains in 4 dimensions, so data reshaping is needed
data = data.reshape(1, 100, 100, 3)
data.shape

# Get the probability that this image is belong to each class
result = model.predict(data)
result.sum() # this is equal to 1

# take the largest probability among 'result'
output = np.argmax(result) 
print(output)




