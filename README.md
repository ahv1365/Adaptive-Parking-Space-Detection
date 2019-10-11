# Contents
1. [Adaptive Parking Space Detection](#adaptive parking space detection)

2. Introduction

3. Convolutional neural networks (CNN)

    3.1. Fully Convolutional Network (FCN)

    3.2. Different Variants of a Region-Based CNN (R-CNN)

      3.2.1. R-CNN

      3.2.2. Fast R-CNN

      3.2.3. Faster R-CNN

4. Object Detection

    4.1. Mask R-CNN

5. Parking Lot Object Detection Implementation

6. Mobile Application

7. Conclusion 

8. Documentation 

# 1. Adaptive Parking Space Detection

This repository focuses on the implementation of the novel object detection regional convolutional network algorithm Mask R-CNN as a system for recognizing the empty spaces in the warehouse parking areas by detecting trucks and cars in the video frames. The process starts with finding and assigning label on the all potential parking spots. This process accomplishes with the help of pre-trained Microsoft COCO dataset and a designed ResNet 101 network on instance segmentation model Mask R-CNN.
  


[<img src="https://i.imgur.com/66qlrLE.jpg" align="center" width="850">](https://flutter.dev/)


# 2. Introduction

In recent years, deep learning and neural networks brought a huge progress in the field of computer vision. Deep convolutional neural networks (CNN) an application of deep learning in computer vision achieved significant success in image classification, semantic segmentation and object detection. Detection, classification and segmentation the objects in images had been always on high demand by companies and applicable in different areas. By applying a system to Control the traffic in a warehouse parking bays, companies are capable to accept more vehicles in the parking areas resulting in costs and waste-time avoidance.
Pretrained algorithms pave the way in designing such an object detection system for developers. By means of these algorithms an instance segmentation model captures the elements in an image and differentiate between the objects. This process works in real-time and generates the data in a pre-adjusted time interval. By these information companies monitors the arrival and departure transit flows of the warehouses.
 

# 3. Convolutional neural networks (CNN)

Convolutional Neural Networks (CNN) are a special type of neural network for processing spatially ordered data. These data include, for example, image information (2 dimensions), videos (3 dimensions) or audio tracks (1-2 dimensions).
Convolutional neural networks also called CNN is an artificial neural network mostly use for image analyzing and also can be used for analyzing data and classification problems. Generally, CNN recognize the patterns precisely and extract some meaningful results from these patterns which makes CNN a useful tool for image analysis.  The difference of CNN and other neural networks such as Multi-layer perception (MLP) is the hidden layers named convolutional layers as well and other non-convolutional layers.  Convolutional layers transfer inputs to other layers with the operation called convolutional operation. 
Filters are the tools by which the patterns are detected which are specified with every convolutional layer. The patterns in an image are edges, texture objects and so on which the filter used for detecting edges is called edge detector or in more advance circles and square detector which are geometric filters. By moving forward in into the network the filters can be more advanced for detecting complex objects such as eye or ear moving towards detecting animals, objects and human. 

# 3.1.	Fully Convolutional Network (FCN)

FCNs are a converted form of convolutional neural network in which the fully connected layer replaced with a 1x1 convolution kernel size make the FCN model benefits from a pre-trained CNN. Fully connected layers in CNN require a certain image size such as AlexNet with image size of 224x224 but for FCNs this limitation will be lift up and input size has no limitation. The paper from Jonathan Long, Evan Shelhamer and Trevor Darrel on "Fully Convolutional Networks for Semantic Segmentation" explains the features of FCN with regards of semantic (pixel wise classification) (6). These features can be described by the way the model works in converting an original image to a semantic segmented image by using different convolution blocks and max pool layers to expand an image to smaller size than its original size. Then class prediction ends up with sampling and deconvolution layers to convert the image to the original size.


[<img src="http://deeplearning.net/tutorial/_images/cat_segmentation.png" align="center" width="850">](https://flutter.dev/)







## Documentation

* [Install Flutter](https://flutter.dev/get-started/)
* [Flutter documentation](https://flutter.dev/docs)
* [Development wiki](https://github.com/flutter/flutter/wiki)
* [Contributing to Flutter](https://github.com/flutter/flutter/blob/master/CONTRIBUTING.md)
