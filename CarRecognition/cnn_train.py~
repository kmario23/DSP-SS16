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

#Shuffle the training/testing data before loading it
def randomize_data(train_file,test_file):
    
    with open(train_file) as f:
        train_list = map(rstrip, f)
    with open(test_file) as f:
        test_list = map(rstrip, f)
    shuffle(train_list)
    shuffle(test_list)
    
    return train_list,test_list

#Since the image dataset contains 44,481 images, we are not able to fit all of them into RAM memory
#Instead we split into chunks, and give these chunks to the stochastic gradient descent
def load_dataset(total_chunks,current_chunk,train_list,test_list,return_train=False):

    dimensions = 3
    #image channel means for the ImageNet dataset
    IMAGE_MEAN = np.array([[103.939],[116.779],[123.68]])
    IMAGE_MEAN = np.resize(IMAGE_MEAN,(3,1,1))
    
    if return_train:
        chunk = np.array_split(train_list,total_chunks)[current_chunk]
        train_images = np.empty([len(chunk),dimensions,224,224])
        filenames = (int(line.split('/')[0]) for line in train_list)    
        train_targets_ = np.array(list(filenames))
        train_targets = np.array_split(train_targets_,total_chunks)[current_chunk]
        train_targets = np.asarray(train_targets)
            
        for x in range(0,len(chunk)):
            image = Image.open(chunk[x], 'r')
            image = np.asarray(image)
            image = image.reshape(dimensions,224,224)
            #image.setflags(write=True)
            #image -= IMAGE_MEAN
            train_images[x] = np.asarray(image)
            train_images[x] = image / np.float32(256)
        return train_images,train_targets - 1            
   
    else:
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
    
    weights_path = 'vgg16_weights.h5'

    #just load the weights up to the fully-connected layers
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')

    #create fully-connected layers with random weights
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

def train_network(train_file,test_file):
    
    #build the vgg16 model
    model = VGG_16()
    #set the optimizer to SGD + nesterov momentum
    sgd = SGD(lr=0.0002, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
    #load the pretrained VGG16 weights
    model.load_weights('report_weights')
    num_epochs = 2
    #since all the training/test data doesn't fit into RAM memory, we split it into 5
    num_data_splits = 5;
    # launch the training loop.
    print("Starting training...")
    # We iterate over epochs
    for epoch in range(num_epochs):
        
        start_time = time.time()
        train_list,test_list = randomize_data(train_file,test_file)
        test_acc = 0
        test_loss = 0
        #go through the training data into stochastic gradient descent fashion
        for x in range(num_data_splits):
            print("Training chunk {} of {}".format(
            x+1, num_data_splits ))
            X_train,Y_train = load_dataset(num_data_splits,x,train_list,test_list,return_train=True)
           
            Y_train = np_utils.to_categorical(Y_train, 281)
            model.fit(X_train, Y_train, batch_size=12, nb_epoch=1,
                      verbose=1)    

            # And a full pass over the test data:
        for x in range(num_data_splits):
            print("Testing chunk {} of {}".format(
            x+1, num_data_splits ))
            x_test,y_test = load_dataset(num_data_splits,x,train_list,test_list,return_train=False)
            y_test = np_utils.to_categorical(y_test, 281)
            score = model.evaluate(x_test, y_test, verbose=0)
            test_acc += score[1]
            test_loss +=score[0]
            #print('Test accuracy:', score[1])
        
        print('Test accuracy:', test_acc / num_data_splits)
        print('Test loss:', test_loss / num_data_splits)
        # Print how much time the current epoch took:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
    #we can save our network weights after the training is done    
    model.save_weights('report_weights')
    

def main():
  
    train_file = 'train_surveillance.txt'
    test_file = 'test_surveillance.txt'
    train_network(train_file,test_file)
    

if __name__ == '__main__':
    main()



