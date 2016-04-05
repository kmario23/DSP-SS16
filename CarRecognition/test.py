import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne

from PIL import Image


def load_dataset():

    train_file = 'train_surveillance.txt'
    test_file = 'test_surveillance.txt'
    dimensions = 3
    temp = open(train_file,'r').read().split('\n')
  
    train_images = np.empty([len(temp)-1,dimensions,96,96])
    f = open(train_file,'r')
    filenames = (int(line.split('/')[0]) for line in f)
    train_targets = np.array(list(filenames))

    for x in range(0,len(temp)-1):
        image = Image.open(temp[x], 'r')
        image = np.asarray(image)
        image = image.reshape(dimensions,96,96)
        train_images[x] = np.asarray(image)
        train_images[x] = image / np.float32(256)

    temp = open(test_file,'r').read().split('\n')

    test_images = np.empty([len(temp)-1,dimensions,96,96])
    f = open(test_file,'r')
    filenames = (int(line.split('/')[0]) for line in f)
    test_targets = np.array(list(filenames))


    for x in range(0,len(temp)-1):
        image = Image.open(temp[x], 'r')
        image = np.asarray(image)
        image = image.reshape(dimensions,96,96)
        test_images[x] = np.asarray(image)
        test_images[x] = image / np.float32(256)

    return train_images,train_targets,test_images,test_targets

def build_cnn(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 3, 96, 96),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=64, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=64, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=512,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 281-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=281,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network

# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def train_network():
     # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    print("Loading dataset..")
    num_epochs = 20;

    x_train,y_train,x_test,y_test = load_dataset()

    print("Building cnn...")
    network = build_cnn(input_var)
    with np.load('model1.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)
    
    #print(type(network))

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates,allow_input_downcast=True)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc],allow_input_downcast=True)   

     # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(x_train, y_train, 100, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the test data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(x_test, y_test, 100, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  test loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  test accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

     # Optionally, you could now dump the network weights to a file like this:
    np.savez('model1.npz', *lasagne.layers.get_all_param_values(network))
    #

def load_test_image(path):
    
    test_image = np.empty([1,3,96,96])
    image = Image.open(path, 'r')
    image = np.asarray(image)
    image = image.reshape(3,96,96)
    image = np.asarray(image)
    image = image / np.float32(256)
    test_image[0] = image;
    return test_image

def test_network(test_data):

    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    network = build_cnn(input_var)
    with np.load('model.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)
    
    print(test_data.shape)
    #for x in range(0,10):
       # indices = np.arange(len(test_data))
       # np.random.shuffle(indices)
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
       # print("Input: ",indices[0])
    predict_fn = theano.function([input_var], T.argmax(test_prediction, axis=1),allow_input_downcast=True)
       # print(predict_fn(test_data[indices[0]].reshape(1,3,96,96)))
    print(predict_fn(test_data[0].reshape(1,3,96,96)))

def main():
   
    #test_image = load_test_image('../test_images/image1.jpg')
    
    #test_network(test_image)
    train_network()


if __name__ == '__main__':
    main()



