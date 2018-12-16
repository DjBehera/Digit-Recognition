import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

####Importing Datset Test and train
dataset  = pd.read_csv('train.csv')
dataset2  = pd.read_csv('test.csv')

#Defining Target set in Training set
y = dataset['label']
#To make a bar_plot for output labels
y.value_counts().plot.bar()

#Converting y label to Dummy labels
y = pd.get_dummies(y,columns = ['label'])

#Defining independent variables in train and test set
X = dataset.drop(['label'],axis = 1)
X_final = dataset2

#To Normalize the data 
X = X / 255
X_final = X_final / 255

# plot some samples
img = X.iloc[0].as_matrix()
img = img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(X.iloc[0,0])
plt.axis("off")
plt.show()

# Reshape
X = X.values.reshape(-1,28,28,1)
X_final = X_final.values.reshape(-1,28,28,1)
print("x_train shape: ",X.shape)
print("test shape: ",X_final.shape)

# Some examples
plt.imshow(X[90][:,:,0],cmap='gray')
plt.show()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 2)

##Applying Convolutional Neural Network 
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.optimizers import RMSprop,Adam
import warnings
warnings.filterwarnings('ignore')
from keras.layers.normalization import BatchNormalization

classifier = Sequential()

classifier.add(Convolution2D(filters = 64,kernel_size = (3,3),padding = 'Same',input_shape = (28,28,1)))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))
classifier.add(Convolution2D(filters = 64,kernel_size = (3,3),padding = 'Same'))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size= (2,2)))
classifier.add(Dropout(0.25))



classifier.add(Convolution2D(filters = 64,kernel_size = (3,3),padding = 'Same'))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))
classifier.add(Convolution2D(filters = 64,kernel_size = (3,3),padding = 'Same'))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.25))

classifier.add(Flatten())

classifier.add(Dense(output_dim = 256))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))
classifier.add(Dropout(0.5))


classifier.add(Dense(output_dim = 10, activation = 'softmax'))


# Define the optimizer
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

classifier.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=0.5,  # randomly rotate images in the range 5 degrees
        zoom_range = 0.5, # Randomly zoom image 5%
        width_shift_range=0.5,  # randomly shift images horizontally 5%
        height_shift_range=0.5,  # randomly shift images vertically 5%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)

# fits the model on batches with real-time data augmentation:
classifier.fit_generator(datagen.flow(X_train, y_train, batch_size=32),validation_data = (X_test,y_test),
                    steps_per_epoch= len(X_train) / 32, epochs=35)

#pred.to_csv('Prediction1.csv', index = False)'''

y_pred = classifier.predict(X_final)

# Plot the loss and accuracy curves for training and validation 


pred = pd.DataFrame({'ImageId': range(1,len(X_final)+1,1),'Label' : 99})

for i in range(len(pred)):
    pred['Label'][i] = np.argmax(y_pred[i])

pred.to_csv('Prediction1.csv', index = False)# -*- coding: utf-8 -*-

