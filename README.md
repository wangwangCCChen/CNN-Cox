# CNN-Cox

A convolutional neural network model for survival prediction based on prognosis-related cascaded Wx feature selection 

Get Started

Example Datasets：
/data/100

Including 7 types of cancer data, 100 gene features were obtained through the Cwx feature selection algorithm.

Training, Validation and Evaluation of CNN-Cox

Trainmodel.py: used to train our CNN-Cox model and 1D-CNNCox model under different numbers of gene signatures for 7 cancer types.
example.ipynb: A running example of 100 gene signatures with Cwx feature selection under 7 cancer types.
wx.py: Cwx feature selection algorithm.
model.py: Our CNN-Cox model and 1D-CNNCox model and NN-Cox model.
