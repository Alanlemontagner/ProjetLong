# ProjetLong

Oceanographic images captured from the northwest coast of Africa are reconstructed by inpainting the missing zones from the existing context. To do so, Deep Learning was implemented with the help of Generative Adversaial Nets to obtain a more accurate representation of the real values. 

the final dataset (after pre-processing) is 1032 images of sizes 64x64 where each pixel represent a degrees celsius value usually around 18-25. The resolution of the pixel is of 1/12 longitudinal (or latitudinal) degree approximately, which is equivalent to 9.25 kilometers.

It is recommended to be familiar with the concepts Convolutional Neural Networks (ConvNets), Autoencoders and Generative Adversarial Networks (GANs).

There are two main scripts:
One of the main codes can be found in 'Inpainting using GANs' on Jupyter Notebook format. The 'keras_adversarial' folder and the 'modelutils.py' script are needed in order to correctly execute it. This file handles the image reconstruction of images that have been artificially manipulated with 'holes' of size 10x10 (single hole) and 6x6 (multiple holes) which represents the zones that were not successfully captured by the satellite due to external perturbations such as the clouds.
The second one ...
