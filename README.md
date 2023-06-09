# Brain_Tumor_Detection_CNN

A brain tumor detection system is a system that will predict whether the given image of the brain has a tumor or not. The system will be used by hospitals to detect the patient’s brain. The doctor will take the MRI of the patient’s brain and will provide the image into the system and the system will then determine whether the brain has a tumor or not.
The system we are building needs to be built with the highest accuracy. Because this system is used by hospitals and a person’s life depends on it.

# Convolutional Neural Networks
  It is a type of machine learning algorithm which is highly used in classification problems. It can be used in both binary classification and multi-classification problems. We will provide an input image to this algorithm and it will classify the image whether the brain has a tumor or not.
  It works in a way that the model has to be pre-trained with images. There are already pre-built functions in the TensorFlow library. We will be using those functions. We can also create neural networks ourselves but it will not be that optimized. Tensorflow already contains functions for creating a neural network that can create neural networks for us within 5-8 lines of code. These neural networks provided by the TensorFlow library are highly optimised. Then, we test it by providing our test data set.

* CNN is created using four layers. Let’s see them one by one.

1. Input layer: This is the first layer in convolutional neural networks. We will pass a 2D image to this layer. The computer sees the image as a rectangle of pixels. We will be using the ImageDataGenerator function of the Keras library which can read them directly from the directory and classify them based on folders.

2. To explain the above, I have created a folder named train which contains the images which we will give to the model while training. It contains two folders. One is ‘no’ which contains all the images of the brain which don’t have tumors. The other folder is ‘yes’ which contains the images of the brain which has a tumor. So the ImageDataGenerator will directly read this data_set folder and will automatically do the work for classifying images.

3. Convolutional Layer: This layer consists of a set of filters. The image passes through these filters and important features of the image are extracted from it. These filters perform some calculations on the input matrix and the result of this calculation is stored in a separate matrix.

4. Pooling layer: This is another layer from which our matrix passes. This layer is added to downsample the number of parameters in the matrix. This prevents the overfitting of the model.
There are two types of POOLING:

MAX_POOLING: In this, we take the maximum value of the patch.
AVG_POOLING: In this, we take the average value of the patch.

5. Dropout Layer: This is an optional layer that you can add to your model. This layer will prevent overfitting by dropping some of the units.

6. Fully Connected Layer: This is the final step in building our neural networks. This layer will flatten the matrix obtained after the pooling layer and the image will be passed through hidden layers. This hidden layer will perform some computations and will finally predict the data.


# Project Prerequisites
The requirement for this project is Python 3.6 and anaconda installed on your computer. I have used Jupyter notebook for this project. You can use whatever you want.
The required modules for this project are –

1. Numpy – pip install numpy
2. Tensorflow – pip install tensorflow
3. Keras – pip install keras
That’s all we need for our Convolutional neural networks. All the functions for creating an optimized Convolutional Neural Network are there in the tensorflow module.
