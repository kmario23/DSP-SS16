import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax
from lasagne.utils import floatX


from PIL import Image
from matplotlib import pyplot as plt
from string import rstrip
import pickle
from random import shuffle

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D,ZeroPadding2D
from keras.utils import np_utils
from keras.optimizers import SGD

import h5py

#Inputs - train_data, test_data
#Outputs - shuffled train_data, shuffled test_data
def randomize_data(test_file):
    
    with open(test_file) as f:
        test_list = map(rstrip, f)
    
    shuffle(test_list)
    
    return test_list

#load test dataset
def load_dataset(total_chunks,current_chunk,test_list):

    dimensions = 3
    #image channel means for the ImageNet dataset
    IMAGE_MEAN = np.array([[103.939],[116.779],[123.68]])
    IMAGE_MEAN = np.resize(IMAGE_MEAN,(3,1,1))
    
    chunk = np.array_split(test_list,total_chunks)[current_chunk]
    test_images = np.empty([len(chunk),dimensions,224,224])
    filenames = (int(line.split('/')[0]) for line in test_list)
    test_targets_ = np.array(list(filenames))
    test_targets = np.array_split(test_targets_,total_chunks)[current_chunk]
    test_targets = np.asarray(test_targets)

    for x in range(0,len(chunk)):
        image = Image.open(chunk[x], 'r')
        image = np.asarray(image)
        image = image.reshape(dimensions,224,224)
        #image.setflags(write=True)
        #image -= IMAGE_MEAN
        test_images[x] = np.asarray(image)
        test_images[x] = image / np.float32(256)

    return test_images,test_targets - 1

#Build the VGG16 architecture (identical to the original paper)
def VGG_16():
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu',trainable = True))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu',trainable = True))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu',trainable = True))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu',trainable = True))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu',trainable = True))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu',trainable = True))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu',trainable = True))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu',trainable = True))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu',trainable = True))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu',trainable = True))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu',trainable = True))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu',trainable = True))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu',trainable = True))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    #create fully-connected layers
    model_m = Sequential()
    model_m.add(model)
    model_m.add(Flatten())
    model_m.add(Dense(4096, activation='relu'))
    model_m.add(Dropout(0.5))
    model_m.add(Dense(4096, activation='relu'))
    model_m.add(Dropout(0.5))
    #we have 281 classes in total
    model_m.add(Dense(281, activation='softmax'))

    return model_m


#test the network with a single image 
def test_image(path):

    model = VGG_16()
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
    model.load_weights('weights_final')

    test_image = np.empty([1,3,224,224])
    image = Image.open(path, 'r')
    image = np.asarray(image)
    image = image.reshape(3,224,224)
    image = np.asarray(image)
    image = image / np.float32(256)
    test_image[0] = image;
    y_pred = model.predict(test_image)
    #output the most probable class
    print(np.argmax(y_pred)+1)

#test the accuracy of the network on the whole test set
def test_network(test_file):

    model = VGG_16()
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
    model.load_weights('weights_final')
    num_data_splits = 5
    test_list = randomize_data(test_file)
    test_acc = 0
    test_loss = 0
    for x in range(num_data_splits):
        print("Testing chunk {} of {}".format(
        x+1, num_data_splits ))
        x_test,y_test = load_dataset(num_data_splits,x,test_list)
        y_test = np_utils.to_categorical(y_test, 281)
        score = model.evaluate(x_test, y_test, verbose=1)
        test_acc += score[1]
        test_loss +=score[0]
    
        
    print('Test accuracy:', test_acc / num_data_splits)
    print('Test loss:', test_loss / num_data_splits)
    

def main():
   
    #test_image('33/dccbbe07e8ff56.jpg')
    
    test_file = 'test_surveillance.txt'
    test_network(test_file)
   

if __name__ == '__main__':
    main()



