#### Table of Contents  
[1. Adaptive Parking Space Detection](#Adaptive)  

[2. Introduction](#Introduction)  

[3. Convolutional neural networks (CNN)](#Convolutional)  

  [3.1. Fully Convolutional Network (FCN)](#Fully) 

  [3.2. Different Variants of a Region-Based CNN (R-CNN)](#Different) 

   [3.2.1. R-CNN](#R-CNN) 

   [3.2.2. Fast R-CNN](#Fast) 
   
   [3.2.3. Faster R-CNN](#Faster)  
   
 [4. Object Detection](#Object)  
 
   [4.1. Mask R-CNN](#Mask)  
 
 [5. Parking Lot Object Detection Implementation](#Parking)  
 
 [6. Mobile Application](#Mobile)  
 
 [7. Conclusion ](#Conclusion)  
 
 [8. Documentation ](#Documentation)  



<a name="Adaptive"/>

# 1. Adaptive Parking Space Detection

This repository focuses on the implementation of the novel object detection regional convolutional network algorithm Mask R-CNN as a system for recognizing the empty spaces in the warehouse parking areas by detecting trucks and cars in the video frames. The process starts with finding and assigning label on the all potential parking spots. This process accomplishes with the help of pre-trained Microsoft COCO dataset and a designed ResNet 101 network on instance segmentation model Mask R-CNN.
  



[<img src="https://i.imgur.com/66qlrLE.jpg" align="center" width="850">](https://github.com/matterport/Mask_RCNN)

<a name="Introduction"/>

# 2. Introduction

In recent years, deep learning and neural networks brought a huge progress in the field of computer vision. Deep convolutional neural networks (CNN) an application of deep learning in computer vision achieved significant success in image classification, semantic segmentation and object detection. Detection, classification and segmentation the objects in images had been always on high demand by companies and applicable in different areas. By applying a system to Control the traffic in a warehouse parking bays, companies are capable to accept more vehicles in the parking areas resulting in costs and waste-time avoidance.
Pretrained algorithms pave the way in designing such an object detection system for developers. By means of these algorithms an instance segmentation model captures the elements in an image and differentiate between the objects. This process works in real-time and generates the data in a pre-adjusted time interval. By these information companies monitors the arrival and departure transit flows of the warehouses.
 
<a name="Convolutional"/>

# 3. Convolutional neural networks (CNN)

Convolutional Neural Networks (CNN) are a special type of neural network for processing spatially ordered data. These data include, for example, image information (2 dimensions), videos (3 dimensions) or audio tracks (1-2 dimensions).
Convolutional neural networks also called CNN is an artificial neural network mostly use for image analyzing and also can be used for analyzing data and classification problems. Generally, CNN recognize the patterns precisely and extract some meaningful results from these patterns which makes CNN a useful tool for image analysis.  The difference of CNN and other neural networks such as Multi-layer perception (MLP) is the hidden layers named convolutional layers as well and other non-convolutional layers.  Convolutional layers transfer inputs to other layers with the operation called convolutional operation. 
Filters are the tools by which the patterns are detected which are specified with every convolutional layer. The patterns in an image are edges, texture objects and so on which the filter used for detecting edges is called edge detector or in more advance circles and square detector which are geometric filters. By moving forward in into the network the filters can be more advanced for detecting complex objects such as eye or ear moving towards detecting animals, objects and human. 

<a name="Fully"/>

# 3.1.	Fully Convolutional Network (FCN)

FCNs are a converted form of convolutional neural network in which the fully connected layer replaced with a 1x1 convolution kernel size make the FCN model benefits from a pre-trained CNN. Fully connected layers in CNN require a certain image size such as AlexNet with image size of 224x224 but for FCNs this limitation will be lift up and input size has no limitation. The paper from Jonathan Long, Evan Shelhamer and Trevor Darrel on "Fully Convolutional Networks for Semantic Segmentation" explains the features of FCN with regards of semantic (pixel wise classification) (6). These features can be described by the way the model works in converting an original image to a semantic segmented image by using different convolution blocks and max pool layers to expand an image to smaller size than its original size. Then class prediction ends up with sampling and deconvolution layers to convert the image to the original size.


[<img src="http://deeplearning.net/tutorial/_images/cat_segmentation.png" align="center" width="850">](https://github.com/matterport/Mask_RCNN)

<a name="Different"/>

# 3.2. Different Variants of a Region-Based CNN (R-CNN)

Creation of region convolutional neural network was to solve the problem of multiple objects detection in an image which with CNN it was computationally overwhelming process. In R-CNN there are two stages of identifying the regions of object (region Proposals) in the image and then classifying the object in the detected region. The three forms of regional object detectors are:

<a name="R-CNN"/>

# 3.2.1.	R-CNN

Region convolutional neural network is a deep learning object detection based on CNN which in the below the structure of the model is illustrated.

[<img src="https://i.imgur.com/k3piqdQ.png" align="center" width="850">](https://github.com/matterport/Mask_RCNN)

<a name="Fast"/>

# 3.2.2.	Fast R-CNN

Fast R-CNN introduced to solve the problems of time-consuming process of training and non-learning of R-CNN algorithm. In compare to R-CNN faster R-CNN process the entire image instead of cropping out and resizing and also by a max pooling CNN layer both feature extraction and ROI input layer are classified.


[<img src="https://i.imgur.com/S2T4MNq.png" align="center" width="850">](https://github.com/matterport/Mask_RCNN)

<a name="Faster"/>

# 3.2.3.	Faster R-CNN

In Faster R-CNN region proposals are predicted by a separated network and then will be reshaped by a ROI pooling layer.

[<img src="https://i.imgur.com/i3NmpQs.png" align="center" width="850">](https://github.com/matterport/Mask_RCNN)

<a name="Object"/>

# 4. Object Detection

A technique in computer vision to identify the location of object instances in the image is called object detection. In order to design an object detection algorithm, it is required to understand the fundamental of image.


<a name="Mask"/>

# 4.1. Mask R-CNN

In the recent years there has happened many progresses in the field of object detection also sematic segmentation as a challenging problem so far. Instant segmentation is a combination of object detection and semantic segmentation (understanding of an image at the pixel level) tries to detect individual objects as well as a mask of each instances and assigning an object class to each pixel in the image.

<a name="Parking"/>

# 5. Parking Lot Object Detection Implementation

In this part the system is tested on a single photo in series to check the accuracy on the pre-adjusted confidence score and check whether confidence score needs to be redefined. The process is already executed on google Collaboratory an online cloud computing system. 


[<img src="https://i.imgur.com/vutJuBu.png" align="center" width="850">](https://github.com/matterport/Mask_RCNN)

<a name="Mobile"/>

# 6. Mobile Application

Mobile application designed for visualizing the results to understand easier and faster the needed data in real time. The app designed for this project has the following features:
•	Detection of empty and occupied places in real-time
•	History records of the detection information
•	Real-time image from the place
•	The detection results of the image


[<img src="https://i.imgur.com/apiaHLO.jpg" align="center" width="850">](https://github.com/matterport/Mask_RCNN)


<center>
<img src="https://i.imgur.com/c3Qmlbw.gif" align="center" width="250">
</center>

<a name="Conclusion"/>

# 7. Conclusion

The system comparison indicates the time required in order an image go through the model and outcome extracts. For the first system model can examine whether in some sequential frames the parking space detected free, but this process takes longer time for the second low performance system to be accomplished. A solution for increasing the speed in the second system was reducing the resolution of the image which were resulted in lower detection accuracy. 
In conclusion, mask R-CNN can also detect the small objects in the photo pixelwise in compare to other object detection models and the accuracy of the system is about 65% with a partially cloudy sky in compared to 26% in a sunny day. 


## Documentation

* [Mask R-CNN model](https://github.com/matterport/Mask_RCNN)

