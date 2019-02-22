## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
![Udacity - Self-Driving Car P3](https://img.shields.io/badge/Status-Pass-green.svg)
# **Traffic Sign Recognition** 
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report/1.png "Visualization"
[image2]: ./report/2.png "Before Grayscaling"
[image3]: ./report/3.png "After Grayscaling"
[image20]: ./report/4.PNG "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./report/5.png "Train Set"
[image6]: ./report/6.png "Validation Set"
[image7]: ./report/7.png "Test Set"
[image8]: ./report/8.png "Acc Graph"
[image14]: ./report/9.png "New Signs Predections"
[image15]: ./report/10.png "Colnolution Layer 1"
[image16]: ./report/11.png "Colnolution Layer 2"
[image17]: ./report/12.png "Colnolution Layer 3"
[image18]: ./report/13.png "Colnolution Layer 4"
[image19]: ./report/14.png "Colnolution Layer 5"

[image10]: ./new_signs/1.jpg "New 1"
[image11]: ./new_signs/2.jpg "New 2"
[image12]: ./new_signs/3.jpg "New 3"
[image13]: ./new_signs/4.jpg "New 4"
[image9]: ./new_signs/5.jpg "New 5"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/lawga/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image5]
![alt text][image6]
![alt text][image7]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because because after visualaization of some random images, I foun that many signs depens on the shape and not on the colour. and also converting to grayscale is good to easily brigthen the images without loosing the qualitry of the image.

After that I used histogram equalization for better visability for the darker images.

Here is an example of a traffic sign image before and after grayscaling and equalisation.

![alt text][image2] ![alt text][image3]

As a last step, I normalized the image data because this will support the convergance of the optimization through the training process.

I decided to generate additional data because adding them synthetically will yield more robust learning to potential deformations in the test set.


To add more data to the the data set, I used the Kera, It Generate batches of tensor image data with real-time data augmentation.
My augmentation to the images was as follows:
rotation range = 12 degrees
zoom range = 0.1x
width shift range = 0.1 pixels
height shift range = 0.1 pixels

Here is an example of an original image and an augmented image:

![alt text][image20]

The difference between the original data set and the augmented data set is that the images are effected by some noise that the test set may or may not have, and adding that noise is good to make the model more robust and overfitting-free.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x55 	|
| RELU					|												|
| Max pooling	      	| 1x1 stride,  outputs 16x16x55 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x89 	|
| RELU					|												|
| Max pooling	      	| 1x1 stride,  outputs 8x8x89 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x144 	|
| RELU					|												|
| Max pooling	      	| 1x1 stride,  outputs 4x4x144 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x89 	|
| RELU					|												|
| Max pooling	      	| 1x1 stride,  outputs 2x2x89 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x55 	|
| RELU					|												|
| Max pooling	      	| 1x1 stride,  outputs 1x1x55 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| (1x1x55)+(2x2x89)+(4x4x144)+(8x8x89)+(16x16x55)        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I followed the same approch as the one mentioned in [Yan LeCun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) neural network architecture.

The different thing about mine is that I used Fibonacci numbers as a refrance of how deep each layer should be. I did some analysis and found that this approch have better results. Maybe I should do more studies on why this is giving better results.

```python
# Define your architecture here.
### Feel free to use as many code cells as needed.
from tensorflow.contrib.layers import flatten
import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True



#0	1	1	2	3	5	8	13	21	34	55	89	144	233	377	
# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.truncated_normal(shape=[3, 3, 1, 55], mean=0.0, stddev=0.05)),
    'wc2': tf.Variable(tf.truncated_normal(shape=[3, 3, 55, 89], mean=0.0, stddev=0.05)),
    'wc3': tf.Variable(tf.truncated_normal(shape=[3, 3, 89, 144], mean=0.0, stddev=0.05)),
    'wc4': tf.Variable(tf.truncated_normal(shape=[3, 3, 144, 89], mean=0.0, stddev=0.05)), #89 was 233
    'wc5': tf.Variable(tf.truncated_normal(shape=[3, 3, 89, 55], mean=0.0, stddev=0.05)), #55 was 377
    'wd1': tf.Variable(tf.truncated_normal(shape=[50], mean=0.0, stddev=0.05)),
    'out': tf.Variable(tf.truncated_normal(shape=[50, n_classes], mean=0.0, stddev=0.05))}



biases = {
    'bc1': tf.Variable(tf.constant(0.05, shape=[55])),
    'bc2': tf.Variable(tf.constant(0.05, shape=[89])),
    'bc3': tf.Variable(tf.constant(0.05, shape=[144])),
    'bc4': tf.Variable(tf.constant(0.05, shape=[89])), #89 was 233
    'bc5': tf.Variable(tf.constant(0.05, shape=[55])),  #55 was 377
    'bd1': tf.Variable(tf.constant(0.05, shape=[50])),
    'out': tf.Variable(tf.constant(0.05, shape=[n_classes]))}

def maxpool2d(x, k=2):
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='SAME')

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def dropouts(input_layer, dropout):
    drop = tf.nn.dropout(input_layer, dropout)
    return drop

def init_W(shape, mu=0, sigma=0.1):
    return tf.Variable(tf.truncated_normal(shape, mean = mu, stddev = sigma))

def init_B(shape, start_val=0.1):
    initialization = tf.constant(start_val, shape=shape)
    return tf.Variable(initialization)


def OBNet(x, weights, biases, dropout):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    #conv# = conv2d(x, W, b, strides=1)
    #drop# = dropout(input_layer, dropout)
    #maxpool2d = maxpool2d(x, k=2)
    #0	1	1	2	3	5	8	13	21	34	55	89	144	233	377	
    ################1st Stage######################
    # Layer 1: Convolutional. Input = 32x32x1. Output = 16x16x55.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    max1 = maxpool2d(conv1, k=2)
    drop1 = dropouts(max1, dropout=keep_prob)
    ################2nd Stage######################
    # Layer 2: Convolutional. Input = 16x16x55. Output = 8x8x89.
    conv2 = conv2d(drop1, weights['wc2'], biases['bc2'])
    max2 = maxpool2d(conv2, k=2)
    drop2 = dropouts(max2, dropout=keep_prob)
    ################3rd Stage######################
    # Layer 3: Convolutional. Input = 8x8x89. Output = 4x4x144.
    conv3 = conv2d(drop2, weights['wc3'], biases['bc3'])
    max3 = maxpool2d(conv3, k=2)
    drop3 = dropouts(max3, dropout=keep_prob)
    ################4th Stage######################
    # Layer 4: Convolutional. Input = 4x4x144. Output = 2x2x89.
    conv4 = conv2d(drop3, weights['wc4'], biases['bc4'])
    max4 = maxpool2d(conv4, k=2)
    drop4 = dropouts(max4, dropout=keep_prob)
    ################5th Stage######################
    # Layer 5: Convolutional. Input = 2x2x89. Output = 1x1x55.
    conv5 = conv2d(drop4, weights['wc5'], biases['bc5'])
    max5 = maxpool2d(conv5, k=2)
    drop5 = dropouts(max5, dropout=keep_prob)
    ################6th Stage######################
    # Layer 6: Fully Connected. Input = (1x1x55)+(2x2x89)+(4x4x144)+(8x8x89)+(16x16x55) Output = 50.
    fc0 = tf.concat([flatten(drop1), flatten(drop2), flatten(drop3), flatten(drop4), flatten(drop5)], 1)
    fc1_W = init_W(shape=(fc0._shape[1].value, 50))
    fc1_B = tf.Variable(tf.constant(0.1, shape=[fc0._shape[1].value]))
    fc1 = tf.matmul(fc0, fc1_W) + biases['bd1']
    drop6 = dropouts(fc1, dropout=keep_prob)
    #################output#######################  
    logits = tf.matmul(drop6, weights['out']) + biases['out'] 
    
    
    # Create a Network parameter dict for visualization
    global network_params
    network_params = {
        "conv1": conv1,
        "conv2": conv2,
        "conv3": conv3,
        "conv4": conv4,
        "conv5": conv5,
        "fc0": fc0,
        "fc1": fc1,
        "logits": logits
    }
                       
    return logits, conv1, conv2, conv3, conv4, conv5
```

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

Training...

EPOCH 1 ...
 |████████████████████████████████████████████████████████████████████████████████████████████████████| 100%
Evaluating...
 |████████████████████████████████████████████████████████████████████████████████████████████████████| 100%
Evaluating...
 |████████████████████████████████████████████████████████████████████████████████████████████████████| 100%
EPOCH 1 ...
Train Accuracy Accuracy = 0.989
Validation Accuracy = 0.962
Model saved
EPOCH 2 ...
 |████████████████████████████████████████████████████████████████████████████████████████████████████| 100%
Evaluating...
 |████████████████████████████████████████████████████████████████████████████████████████████████████| 100%
Evaluating...
 |████████████████████████████████████████████████████████████████████████████████████████████████████| 100%
EPOCH 2 ...
Train Accuracy Accuracy = 0.996
Validation Accuracy = 0.976
Model saved
EPOCH 3 ...
 |████████████████████████████████████████████████████████████████████████████████████████████████████| 100%
Evaluating...
 |████████████████████████████████████████████████████████████████████████████████████████████████████| 100%
Evaluating...
 |████████████████████████████████████████████████████████████████████████████████████████████████████| 100%
EPOCH 3 ...
Train Accuracy Accuracy = 0.996
Validation Accuracy = 0.978
Model saved
EPOCH 4 ...
 |████████████████████████████████████████████████████████████████████████████████████████████████████| 100%
Evaluating...
 |████████████████████████████████████████████████████████████████████████████████████████████████████| 100%
Evaluating...
 |████████████████████████████████████████████████████████████████████████████████████████████████████| 100%
EPOCH 4 ...
Train Accuracy Accuracy = 0.998
Validation Accuracy = 0.980
Model saved
EPOCH 5 ...
 |████████████████████████████████████████████████████████████████████████████████████████████████████| 100%
Evaluating...
 |████████████████████████████████████████████████████████████████████████████████████████████████████| 100%
Evaluating...
 |████████████████████████████████████████████████████████████████████████████████████████████████████| 100%
EPOCH 5 ...
Train Accuracy Accuracy = 0.998
Validation Accuracy = 0.986
Model saved
EPOCH 6 ...
 |████████████████████████████████████████████████████████████████████████████████████████████████████| 100%
Evaluating...
 |████████████████████████████████████████████████████████████████████████████████████████████████████| 100%
Evaluating...
 |████████████████████████████████████████████████████████████████████████████████████████████████████| 100%
EPOCH 6 ...
Train Accuracy Accuracy = 0.999
Validation Accuracy = 0.984
Model saved
EPOCH 7 ...
 |████████████████████████████████████████████████████████████████████████████████████████████████████| 100%
Evaluating...
 |████████████████████████████████████████████████████████████████████████████████████████████████████| 100%
Evaluating...
 |████████████████████████████████████████████████████████████████████████████████████████████████████| 100%
EPOCH 7 ...
Train Accuracy Accuracy = 0.999
Validation Accuracy = 0.983
Model saved
EPOCH 8 ...
 |████████████████████████████████████████████████████████████████████████████████████████████████████| 100%
Evaluating...
 |████████████████████████████████████████████████████████████████████████████████████████████████████| 100%
Evaluating...
 |████████████████████████████████████████████████████████████████████████████████████████████████████| 100%
EPOCH 8 ...
Train Accuracy Accuracy = 0.999
Validation Accuracy = 0.983
Model saved
EPOCH 9 ...
 |████████████████████████████████████████████████████████████████████████████████████████████████████| 100%
Evaluating...
 |████████████████████████████████████████████████████████████████████████████████████████████████████| 100%
Evaluating...
 |████████████████████████████████████████████████████████████████████████████████████████████████████| 100%
EPOCH 9 ...
Train Accuracy Accuracy = 0.998
Validation Accuracy = 0.985




INFO:tensorflow:Restoring parameters from ./obnet
Evaluating...
 |████████████████████████████████████████████████████████████████████████████████████████████████████| 100%
Performance on test set: 0.966

My final model results were:
* training set accuracy of 99.8%
* validation set accuracy of 98.5% 
* test set accuracy of 96.6% 

![alt text][image8]

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?  Yan LeCun neural network architecture [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) was the first architecture which led to low accuarcey on both validation and test sets(Underfitting). Then evolved into a completly different one that has some of the base of the DeepNet and [Yan LeCun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) neural network architecture.

* What were some problems with the initial architecture? passing a coloured images without augmanting them or passing them throug any kinf of filteration made the model more biased to some signs that I found later on that they have more apparance in the data sets.
* How was the architecture adjusted and why was it adjusted? I made 5 Convolutional layers and put a ppoling layer then a dropout between each one of them and then concatnated them in a one fully layer that then transformed to a smaller layer that produced the output.

* Which parameters were tuned? How were they adjusted and why?
* EPOCHS : Earlier when I used Lenet arcutecture, I had to do 30 epochs, but with the new network I came up with, I only use 5
* BATCH_SIZE : the size of the batch depends on the memory, and becuse I used Udacity Cloud, The size did not really matter.
* BATCHES_PER_EPOCH : I had to do some trails to know what best for my network, but at the end I understood how this works on tensorflow and decided that 5000
* What are some of the important design choices and why were they chosen? The initialization of the weights with small vluse definitly helped the mode with reaching a higer accurace at very low time. 
How might a dropout layer help with creating a successful model? the Newwork I have has 5 layes of Convnets, which go up to 144  lavels in one of them, the dropouts help with the speed and also keeping the the network from overfitting the training set. I used a Fibonaci numbers approch in deciding what the depth of each Convolution Layer should be.

If a well known architecture was chosen:
* What architecture was chosen? ConvNets
* Why did you believe it would be relevant to the traffic sign application?The architecture used in the present work departs from traditional ConvNets by the type of non-linearities used, by the use of connections that skip layers, and by the use of pooling layers with different subsampling ratios for the connections that skip layers and for those that do not. it is based on [Yan LeCun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) paper, whwre they tested a ConvNet network on traffic Signs. So I took there work and improved on it to some level of my understanding and to the limits of my imagination.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? Without looking at the Valiodation set and the test set, the Model was able to reach 98% and 96% respectively. 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image9] ![alt text][image10] ![alt text][image11]
![alt text][image12] ![alt text][image13]

The first, Seoned and last images might be difficult to classify because they were not interduced in the training set. So I was confidant that those images will not be recognized.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Small Motors Only      		| Right-of-way at the next intersection 									| 
| Walk & Bicyles Only     			| Road work 										|
| 30 km/h					| 30 km/h											|
| Dangerous curve to the right	      		| Dangerous curve to the right					 				|
| Street Narrows Down			| General caution      							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This compares favorably to the accuracy on the test set of 96% Knowing that uninterduced signs will not be recognised.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.94), and the image does contain a stop sign. The top five soft max probabilities were for each sign as follows:

INFO:tensorflow:Restoring parameters from ./obnet
TopKV2(values=array([[  9.41752434e-01,   5.07599749e-02,   5.67284878e-03,
          8.28838907e-04,   2.89341871e-04],
       [  6.43326700e-01,   2.04405591e-01,   1.33099422e-01,
          3.21466755e-03,   3.07182339e-03],
       [  9.99999285e-01,   6.56961561e-07,   4.97985617e-08,
          4.53239473e-08,   4.96465091e-09],
       [  9.99999642e-01,   3.65144160e-07,   1.27237720e-10,
          7.59404258e-12,   7.33098911e-12],
       [  9.99908566e-01,   8.88587529e-05,   2.13025078e-06,
          2.24513954e-07,   1.90187052e-07]], dtype=float32), indices=array([[11, 21, 40, 30, 31],
       [25, 12, 30, 21, 11],
       [ 1,  2,  0,  5,  7],
       [20, 23, 19, 11, 41],
       [18, 27, 11, 24, 26]], dtype=int32))

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .94         			| Right-of-way at the next intersection   									| 
| .64     				| Road work 										|
| .99					| 30 km/h											|
| .99	      			| Dangerous curve to the right					 				|
| .99				    | General caution     							|

![alt text][image14]


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Each layer Showed different properties that it was looking for as follows:

Convolution Layer #1:
![alt text][image15]

Convolution Layer #2:
![alt text][image16]

Convolution Layer #3:
![alt text][image17]

Convolution Layer #4:
![alt text][image18]

Convolution Layer #5:
![alt text][image19]
