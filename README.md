# road_classifier
Image segmentation for detecting the road/ground plane

Remanents of a previous project for learning Tensorflow. The goal was to segment the road in images from an onbaord (front facing) camera. All labeled images were binary masks (i.e. no multi-class segmentation). 

A variety of models exist in the testing_models folder:
* Variation of [Fully Convolutional Network](https://arxiv.org/pdf/1605.06211.pdf)
* Variation of [E-Net](https://arxiv.org/abs/1606.02147)
* A series of incrementally larger architectures for testing the influence of depth, down/upsampling, skip connections, bottleneck units, etc.

The architecture in model.py is by no means the best architecture, it is simply one that performed decently with reasonable training time.
