Multi layer NN from scratch
===============

**Machine Learning Course Project**

**Authors:**
Cignoni Giacomo, 
Marinelli Alberto Roberto, 
Melero Cavallo Martina

 Master Degree Computer Science, AI curriculum ML course (654AA),
 Academic Year: 2021-2022

 

 This repository contains the implementation from scratch (using Python and Numpy) of an Artificial Neural Network able
 to solve both classification and regression problems. Although it
 works on different datasets, in our case it was tested on the Monk
 classification problem and ML course's CUP (regression).

The objective of this project was to
 find the best configurations of our Neural Network implementation for both of these
 problems.

 We implemented the building, training, K-fold cross
 validation and testing of a feed-forward Neural Network (with momentum
 and L2 regularization) using back-propagation. 
The algorithm is able
 to run on on-line, mini-batch and batch configurations and perform
 grid-search to find the best set of hyper-parameters, i.e. with a low
 and smooth loss curve.

 As extensions, we coded the linear learning
 rate decay, L1 regularization, the possibility to have an arbitrary
 number of hidden layers and also the option to pursue a randomized
 Neural Network and ensemble approach.
 
 More infos in the report.pdf file.
