Model Architecture
==================

The Portal Dose prediction model is based on a convolutional U-Net architecture.

The network receives a preprocessed transit EPID image as input and directly
predicts the corresponding two-dimensional Portal Dose distribution.

.. image:: images/architecture.png
   :width: 700px
   :align: center

An ensemble of independently trained models is used during inference to improve
prediction robustness and reduce variance.