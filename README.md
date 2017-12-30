# Semantic Segmentation

[//]: # (Image References)

[image00]: ./images/loss.png "Error Loss"
[image01]: ./images/sample01.png "Sample 01"
[image02]: ./images/sample02.png "Sample 02"
[image03]: ./images/sample03.png "Sample 03"
[image04]: ./images/sample04.png "Sample 04"
[image05]: ./images/sample05.png "Sample 05"
[image06]: ./images/sample06.png "Sample 06"
[image11]: ./images/sample11.png "Sample 11"
[image12]: ./images/sample12.png "Sample 12"
[image13]: ./images/sample13.png "Sample 13"
[image14]: ./images/sample14.png "Sample 14"
[image15]: ./images/sample15.png "Sample 15"
[image16]: ./images/sample16.png "Sample 16"



### Introduction

This project aims at constructing a fully convolutional neural network for performing semantic segmentation to identify drivable road from a car dashcam image (trained and tested on the KITTI data set). 

### Architecture

The architecture of the fully convolutional neural network is based on the VGG-16 image classifier. The final fully connected layer, Layer 7, of VGG-16 image classifier is converted into a 1x1 convolution and the depth of two, road and not-road, is set. Layer 3 and 4 are also converted similarly and added to Layer 7 as skip connections after it is decoded by upsampling/transposing it. Regularization is applied to each convolutional and transposed convolutional layer.

The hyperparameters used for training are

* keep probability: 0.8
* learning rate: 0.001
* epochs: 90
* batch size: 5
* regularization scalar: 0.001


### Results

The following figure shows transitions of average error losses for each epoch. The blue line shows the one when the output of pooling layers, Layer 3 and Layer 4, are scaled before those layers are added to Layer 7, and the orange line shows the one when the output of those layers are not scaled before those layers are added to Layer 7.

In both of the cases, the average error losses are decreasing over over time. Although the blue line is slower to decrease the average loss than the orange line, it reaches a slightly better result than the other with respect to the average error loss.

![alt text][image00] <!-- .element height="450" width="650" -->

#### Samples

The following images show some results of inference of the trained network. The top one is from the network with scaled pooling output and the bottom one is from the network without scaled pooling output.

One observation is that the network with scaled pooling output produces better results under a difficult situation, e.g. there are shadows, lanes are unclear, and the network without scaled pooling output produce clearer segmentation when other objects such as vehicles are included.

---

![alt text][image01] <!-- .element height="100" width="200" -->
![alt text][image11] <!-- .element height="100" width="200" -->

---

![alt text][image02] <!-- .element height="100" width="200" -->
![alt text][image12] <!-- .element height="100" width="200" -->

---

![alt text][image03] <!-- .element height="100" width="200" -->
![alt text][image13] <!-- .element height="100" width="200" -->

---

![alt text][image04] <!-- .element height="100" width="200" -->
![alt text][image14] <!-- .element height="100" width="200" -->

---

![alt text][image05] <!-- .element height="100" width="200" -->
![alt text][image15] <!-- .element height="100" width="200" -->

---

![alt text][image06] <!-- .element height="100" width="200" -->
![alt text][image16] <!-- .element height="100" width="200" -->


---
## *The following is the original Udacity README*

### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).
