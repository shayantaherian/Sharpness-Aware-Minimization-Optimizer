# Sharpness-Aware-Minimization

This repository presents a potrch implementation of image classification using sharpness-aware minimization (SAM) optimizer. SAM simultaneously minimizes loss value and loss sharpness. In particular, it seeks parameters that lie in neighborhoods having uniformly low loss which results in improving generalization capability. This is an unofficial implementation of [Sharpness-Aware Minimization for Efficiently Improving
Generalization
](https://arxiv.org/pdf/2010.01412.pdf). In this repository the SAM optimizer has been adopted from the above paper and ResNet model has been chosen for trainig the classifier on CIFAR10 dataset. It is noted that the simulation results compared SAM optimizer with Adam and SGD optimizer to demonstrates the superiority of SAM respect to others. The comparison of three optimization can be seen as follows:

<p align="center">
<img src="https://user-images.githubusercontent.com/51369142/116109471-b79c6d80-a6ac-11eb-899d-c98169f5ab49.PNG" width="500" height="200">
 </p>
 
 
