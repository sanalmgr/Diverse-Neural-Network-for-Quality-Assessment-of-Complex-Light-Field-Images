# Diverse-Neural-Network-for-Quality-Assessment-of-Complex-Light-Field-Images
In this work, we propose a LF-IQA method that accurately predicts the quality of complex LF images in accordance with  corresponding subjective quality scores. Our methods is based on a diverse neural network that includes Convolutional Neural Network (CNN), Atrous Convolutional Layer (ACL) with three variants of Atrous rates, and Long Short-Term Memory (LSTM) network layers. More specifically, the model architecture is composed of two streams, each containing CNN, ACL, and LSTM layers that extract spatial and angular features from the horizontal and vertical epipolar plane images. The feature vectors obtained from each stream, are concatenated and fed to the regression block for quality prediction.

## Code:
## Training Model:
1. Prepare the horizontal and vertical EPIs using the method MultiEPL https://bit.ly/3Da8fB6.
2. To train the model, import functions from training_model.py file, and pass the parameters accordingly.
