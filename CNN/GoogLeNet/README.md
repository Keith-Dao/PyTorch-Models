# GoogLeNet

Implementation of GoogLeNet/InceptionNet-V1 as described in the paper.

## MNIST

### Model summary

|    Layer (type)     |    Output Shape    |  Param #  |
| :-----------------: | :----------------: | :-------: |
|   2D convolution    | [-1, 64, 112, 112] |   9,472   |
|        ReLU         | [-1, 64, 112, 112] |     0     |
|   2D max pooling    |  [-1, 64, 56, 56]  |     0     |
| Local response norm |  [-1, 64, 56, 56]  |     0     |
|   2D convolution    |  [-1, 64, 56, 56]  |   4,160   |
|        ReLU         |  [-1, 64, 56, 56]  |     0     |
|   2D convolution    | [-1, 192, 56, 56]  |  110,784  |
|        ReLU         | [-1, 192, 56, 56]  |     0     |
| Local response norm | [-1, 192, 56, 56]  |     0     |
|   2D max pooling    | [-1, 192, 28, 28]  |     0     |
|  **inception_3a**   |        ---         |    ---    |
|   2D convolution    |  [-1, 64, 28, 28]  |  12,352   |
|        ReLU         |  [-1, 64, 28, 28]  |     0     |
|   2D convolution    |  [-1, 96, 28, 28]  |  18,528   |
|        ReLU         |  [-1, 96, 28, 28]  |     0     |
|   2D convolution    | [-1, 128, 28, 28]  |  110,720  |
|        ReLU         | [-1, 128, 28, 28]  |     0     |
|   2D convolution    |  [-1, 16, 28, 28]  |   3,088   |
|        ReLU         |  [-1, 16, 28, 28]  |     0     |
|   2D convolution    |  [-1, 32, 28, 28]  |  12,832   |
|        ReLU         |  [-1, 32, 28, 28]  |     0     |
|   2D max pooling    | [-1, 192, 28, 28]  |     0     |
|   2D convolution    |  [-1, 32, 28, 28]  |   6,176   |
|        ReLU         |  [-1, 32, 28, 28]  |     0     |
|   Inception block   | [-1, 256, 28, 28]  |     0     |
|  **inception_3b**   |        ---         |    ---    |
|   2D convolution    | [-1, 128, 28, 28]  |  32,896   |
|        ReLU         | [-1, 128, 28, 28]  |     0     |
|   2D convolution    | [-1, 128, 28, 28]  |  32,896   |
|        ReLU         | [-1, 128, 28, 28]  |     0     |
|   2D convolution    | [-1, 192, 28, 28]  |  221,376  |
|        ReLU         | [-1, 192, 28, 28]  |     0     |
|   2D convolution    |  [-1, 32, 28, 28]  |   8,224   |
|        ReLU         |  [-1, 32, 28, 28]  |     0     |
|   2D convolution    |  [-1, 96, 28, 28]  |  76,896   |
|        ReLU         |  [-1, 96, 28, 28]  |     0     |
|   2D max pooling    | [-1, 256, 28, 28]  |     0     |
|   2D convolution    |  [-1, 64, 28, 28]  |  16,448   |
|        ReLU         |  [-1, 64, 28, 28]  |     0     |
|   Inception block   | [-1, 480, 28, 28]  |     0     |
|   2D max pooling    | [-1, 480, 14, 14]  |     0     |
|  **inception_4a**   |        ---         |    ---    |
|   2D convolution    | [-1, 192, 14, 14]  |  92,352   |
|        ReLU         | [-1, 192, 14, 14]  |     0     |
|   2D convolution    |  [-1, 96, 14, 14]  |  46,176   |
|        ReLU         |  [-1, 96, 14, 14]  |     0     |
|   2D convolution    | [-1, 208, 14, 14]  |  179,920  |
|        ReLU         | [-1, 208, 14, 14]  |     0     |
|   2D convolution    |  [-1, 16, 14, 14]  |   7,696   |
|        ReLU         |  [-1, 16, 14, 14]  |     0     |
|   2D convolution    |  [-1, 48, 14, 14]  |  19,248   |
|        ReLU         |  [-1, 48, 14, 14]  |     0     |
|   2D max pooling    | [-1, 480, 14, 14]  |     0     |
|   2D convolution    |  [-1, 64, 14, 14]  |  30,784   |
|        ReLU         |  [-1, 64, 14, 14]  |     0     |
|   Inception block   | [-1, 512, 14, 14]  |     0     |
|      **aux_1**      |        ---         |    ---    |
| 2D average pooling  |  [-1, 512, 4, 4]   |     0     |
|   2D convolution    |  [-1, 128, 4, 4]   |  65,664   |
|        ReLU         |  [-1, 128, 4, 4]   |     0     |
| Flatten [-1, 2048]  |         0          |
|  Linear [-1, 1024]  |     2,098,176      |
|        ReLU         |     [-1, 1024]     |     0     |
|       Dropout       |     [-1, 1024]     |     0     |
|   Linear [-1, 10]   |       10,250       |
|  **inception_4b**   |        ---         |    ---    |
|   2D convolution    | [-1, 160, 14, 14]  |  82,080   |
|        ReLU         | [-1, 160, 14, 14]  |     0     |
|   2D convolution    | [-1, 112, 14, 14]  |  57,456   |
|        ReLU         | [-1, 112, 14, 14]  |     0     |
|   2D convolution    | [-1, 224, 14, 14]  |  226,016  |
|        ReLU         | [-1, 224, 14, 14]  |     0     |
|   2D convolution    |  [-1, 24, 14, 14]  |  12,312   |
|        ReLU         |  [-1, 24, 14, 14]  |     0     |
|   2D convolution    |  [-1, 64, 14, 14]  |  38,464   |
|        ReLU         |  [-1, 64, 14, 14]  |     0     |
|   2D max pooling    | [-1, 512, 14, 14]  |     0     |
|   2D convolution    |  [-1, 64, 14, 14]  |  32,832   |
|        ReLU         |  [-1, 64, 14, 14]  |     0     |
|   Inception block   | [-1, 512, 14, 14]  |     0     |
|  **inception_4c**   |        ---         |    ---    |
|   2D convolution    | [-1, 128, 14, 14]  |  65,664   |
|        ReLU         | [-1, 128, 14, 14]  |     0     |
|   2D convolution    | [-1, 128, 14, 14]  |  65,664   |
|        ReLU         | [-1, 128, 14, 14]  |     0     |
|   2D convolution    | [-1, 256, 14, 14]  |  295,168  |
|        ReLU         | [-1, 256, 14, 14]  |     0     |
|   2D convolution    |  [-1, 24, 14, 14]  |  12,312   |
|        ReLU         |  [-1, 24, 14, 14]  |     0     |
|   2D convolution    |  [-1, 64, 14, 14]  |  38,464   |
|        ReLU         |  [-1, 64, 14, 14]  |     0     |
|   2D max pooling    | [-1, 512, 14, 14]  |     0     |
|   2D convolution    |  [-1, 64, 14, 14]  |  32,832   |
|        ReLU         |  [-1, 64, 14, 14]  |     0     |
|   Inception block   | [-1, 512, 14, 14]  |     0     |
|  **inception_4d**   |        ---         |    ---    |
|   2D convolution    | [-1, 112, 14, 14]  |  57,456   |
|        ReLU         | [-1, 112, 14, 14]  |     0     |
|   2D convolution    | [-1, 144, 14, 14]  |  73,872   |
|        ReLU         | [-1, 144, 14, 14]  |     0     |
|   2D convolution    | [-1, 288, 14, 14]  |  373,536  |
|        ReLU         | [-1, 288, 14, 14]  |     0     |
|   2D convolution    |  [-1, 32, 14, 14]  |  16,416   |
|        ReLU         |  [-1, 32, 14, 14]  |     0     |
|   2D convolution    |  [-1, 64, 14, 14]  |  51,264   |
|        ReLU         |  [-1, 64, 14, 14]  |     0     |
|   2D max pooling    | [-1, 512, 14, 14]  |     0     |
|   2D convolution    |  [-1, 64, 14, 14]  |  32,832   |
|        ReLU         |  [-1, 64, 14, 14]  |     0     |
|   Inception block   | [-1, 528, 14, 14]  |     0     |
|      **aux_2**      |        ---         |    ---    |
| 2D average pooling  |  [-1, 528, 4, 4]   |     0     |
|   2D convolution    |  [-1, 128, 4, 4]   |  67,712   |
|        ReLU         |  [-1, 128, 4, 4]   |     0     |
|       Flatten       |     [-1, 2048]     |     0     |
|       Linear        |     [-1, 1024]     | 2,098,176 |
|        ReLU         |     [-1, 1024]     |     0     |
|       Dropout       |     [-1, 1024]     |     0     |
|       Linear        |      [-1, 10]      |  10,250   |
|  **inception_4e**   |        ---         |    ---    |
|   2D convolution    | [-1, 256, 14, 14]  |  135,424  |
|        ReLU         | [-1, 256, 14, 14]  |     0     |
|   2D convolution    | [-1, 160, 14, 14]  |  84,640   |
|        ReLU         | [-1, 160, 14, 14]  |     0     |
|   2D convolution    | [-1, 320, 14, 14]  |  461,120  |
|        ReLU         | [-1, 320, 14, 14]  |     0     |
|   2D convolution    |  [-1, 32, 14, 14]  |  16,928   |
|        ReLU         |  [-1, 32, 14, 14]  |     0     |
|   2D convolution    | [-1, 128, 14, 14]  |  102,528  |
|        ReLU         | [-1, 128, 14, 14]  |     0     |
|   2D max pooling    | [-1, 528, 14, 14]  |     0     |
|   2D convolution    | [-1, 128, 14, 14]  |  67,712   |
|        ReLU         | [-1, 128, 14, 14]  |     0     |
|   Inception block   | [-1, 832, 14, 14]  |     0     |
|   2D max pooling    |  [-1, 832, 7, 7]   |     0     |
|  **inception_5a**   |        ---         |    ---    |
|   2D convolution    |  [-1, 256, 7, 7]   |  213,248  |
|        ReLU         |  [-1, 256, 7, 7]   |     0     |
|   2D convolution    |  [-1, 160, 7, 7]   |  133,280  |
|        ReLU         |  [-1, 160, 7, 7]   |     0     |
|   2D convolution    |  [-1, 320, 7, 7]   |  461,120  |
|        ReLU         |  [-1, 320, 7, 7]   |     0     |
|   2D convolution    |   [-1, 32, 7, 7]   |  26,656   |
|        ReLU         |   [-1, 32, 7, 7]   |     0     |
|   2D convolution    |  [-1, 128, 7, 7]   |  102,528  |
|        ReLU         |  [-1, 128, 7, 7]   |     0     |
|   2D max pooling    |  [-1, 832, 7, 7]   |     0     |
|   2D convolution    |  [-1, 128, 7, 7]   |  106,624  |
|        ReLU         |  [-1, 128, 7, 7]   |     0     |
|   Inception block   |  [-1, 832, 7, 7]   |     0     |
|  **inception_5b**   |        ---         |    ---    |
|   2D convolution    |  [-1, 384, 7, 7]   |  319,872  |
|        ReLU         |  [-1, 384, 7, 7]   |     0     |
|   2D convolution    |  [-1, 192, 7, 7]   |  159,936  |
|        ReLU         |  [-1, 192, 7, 7]   |     0     |
|   2D convolution    |  [-1, 384, 7, 7]   |  663,936  |
|        ReLU         |  [-1, 384, 7, 7]   |     0     |
|   2D convolution    |   [-1, 48, 7, 7]   |  39,984   |
|        ReLU         |   [-1, 48, 7, 7]   |     0     |
|   2D convolution    |  [-1, 128, 7, 7]   |  153,728  |
|        ReLU         |  [-1, 128, 7, 7]   |     0     |
|   2D max pooling    |  [-1, 832, 7, 7]   |     0     |
|   2D convolution    |  [-1, 128, 7, 7]   |  106,624  |
|        ReLU         |  [-1, 128, 7, 7]   |     0     |
|   Inception block   |  [-1, 1024, 7, 7]  |     0     |
| 2D average pooling  |  [-1, 1024, 1, 1]  |     0     |
|       Flatten       |     [-1, 1024]     |     0     |
|       Dropout       |     [-1, 1024]     |     0     |
|       Linear        |      [-1, 10]      |  10,250   |

|                                         |            |
| --------------------------------------- | ---------- |
| Total params                            | 10,334,030 |
| Trainable params                        | 10,334,030 |
| Total params excluding auxiliary params | 5,983,802  |
| Non-trainable params                    | 0          |

### Results

Trained over 20 epochs with a learning rate of 1e-4, batch size of 128 and auxiliary loss weighting of 0.3.

|                     Loss                      |                       Accuracy                        |
| :-------------------------------------------: | :---------------------------------------------------: |
| ![MNIST loss graph](Resources/mnist_loss.png) | ![MNIST accuracy graph](Resources/mnist_accuracy.png) |

|              | Training | Validation | Testing |
| :----------: | :------: | :--------: | :-----: |
|     Loss     |  0.0365  |   0.0337   | 0.0268  |
| Accuracy (%) |  99.32   |   98.96    |  99.14  |

| Class | Training Precision | Validation Precision | Testing Precision | Training Recall | Validation Recall | Testing Recall | Training F1 Score | Validation F1 Score | Testing F1 Score |
| :---: | :----------------: | :------------------: | :---------------: | :-------------: | :---------------: | :------------: | :---------------: | :-----------------: | :--------------: |
|   0   |       0.9949       |        0.9978        |      0.9949       |     0.9954      |      0.9939       |     0.9969     |      0.9952       |       0.9958        |      0.9959      |
|   1   |       0.9957       |        0.9883        |      0.9886       |     0.9953      |      0.9941       |     0.9974     |      0.9955       |       0.9912        |      0.9930      |
|   2   |       0.9931       |        0.9885        |      0.9885       |     0.9941      |      0.9908       |     0.9981     |      0.9936       |       0.9897        |      0.9932      |
|   3   |       0.9958       |        0.9973        |      1.0000       |     0.9934      |      0.9750       |     0.9822     |      0.9946       |       0.9860        |      0.9910      |
|   4   |       0.9929       |        0.9882        |      0.9919       |     0.9919      |      0.9932       |     0.9949     |      0.9924       |       0.9907        |      0.9934      |
|   5   |       0.9932       |        0.9762        |      0.9706       |     0.9913      |      0.9932       |     0.9989     |      0.9923       |       0.9846        |      0.9845      |
|   6   |       0.9933       |        0.9931        |      0.9968       |     0.9935      |      0.9919       |     0.9791     |      0.9934       |       0.9925        |      0.9879      |
|   7   |       0.9899       |        0.9912        |      0.9941       |     0.9928      |      0.9886       |     0.9854     |      0.9914       |       0.9899        |      0.9897      |
|   8   |       0.9937       |        0.9823        |      0.9928       |     0.9942      |      0.9908       |     0.9949     |      0.9939       |       0.9865        |      0.9938      |
|   9   |       0.9893       |        0.9914        |      0.9950       |     0.9895      |      0.9846       |     0.9861     |      0.9894       |       0.9880        |      0.9905      |

From the loss graph, it can be seen that the training and validation loss converges to similar values. Thus, the model appears to have reach its capacity when training for MNIST. That being said, the model performs well given the number of parameters used for inferencing. Note that the training loss includes the weighted loss of the auxiliary classifiers, which may cause the training loss to be larger than the validation loss while being close in other metrics.

## References

Research paper: https://arxiv.org/pdf/1409.4842.pdf
