# ResNet

Implementation of ResNet as described in the paper.

For the purposes of analysis, only the ResNet-50 model would be trained, but the ResNet-101 and ResNet-152 models have been implemented.

## MNIST

### Model summary

|             Layer (type)             |    Output Shape    |  Param #  |
| :----------------------------------: | :----------------: | :-------: |
|              **conv1**               |                    |           |
|            2D convolution            | [-1, 64, 112, 112] |   9,408   |
|            2D batch norm             | [-1, 64, 112, 112] |    128    |
|                 ReLU                 | [-1, 64, 112, 112] |     0     |
|             **conv2_x**              |        ---         |    ---    |
|            2D max pooling            |  [-1, 64, 56, 56]  |     0     |
|            2D convolution            |  [-1, 64, 56, 56]  |   4,096   |
|            2D batch norm             |  [-1, 64, 56, 56]  |    128    |
|                 ReLU                 |  [-1, 64, 56, 56]  |     0     |
|            2D convolution            |  [-1, 64, 56, 56]  |  36,864   |
|            2D batch norm             |  [-1, 64, 56, 56]  |    128    |
|                 ReLU                 |  [-1, 64, 56, 56]  |     0     |
|            2D convolution            | [-1, 256, 56, 56]  |  16,384   |
|            2D batch norm             | [-1, 256, 56, 56]  |    512    |
| 2D convolution (Residual projection) | [-1, 256, 56, 56]  |  16,384   |
| 2D batch norm (Residual projection)  | [-1, 256, 56, 56]  |    512    |
|     Add (Bottleneck + residual)      | [-1, 256, 56, 56]  |     0     |
|                 ReLU                 | [-1, 256, 56, 56]  |     0     |
|            2D convolution            |  [-1, 64, 56, 56]  |  16,384   |
|            2D batch norm             |  [-1, 64, 56, 56]  |    128    |
|                 ReLU                 |  [-1, 64, 56, 56]  |     0     |
|            2D convolution            |  [-1, 64, 56, 56]  |  36,864   |
|            2D batch norm             |  [-1, 64, 56, 56]  |    128    |
|                 ReLU                 |  [-1, 64, 56, 56]  |     0     |
|            2D convolution            | [-1, 256, 56, 56]  |  16,384   |
|            2D batch norm             | [-1, 256, 56, 56]  |    512    |
|               Identity               | [-1, 256, 56, 56]  |     0     |
|     Add (Bottleneck + residual)      | [-1, 256, 56, 56]  |     0     |
|                 ReLU                 | [-1, 256, 56, 56]  |     0     |
|            2D convolution            |  [-1, 64, 56, 56]  |  16,384   |
|            2D batch norm             |  [-1, 64, 56, 56]  |    128    |
|                 ReLU                 |  [-1, 64, 56, 56]  |     0     |
|            2D convolution            |  [-1, 64, 56, 56]  |  36,864   |
|            2D batch norm             |  [-1, 64, 56, 56]  |    128    |
|                 ReLU                 |  [-1, 64, 56, 56]  |     0     |
|            2D convolution            | [-1, 256, 56, 56]  |  16,384   |
|            2D batch norm             | [-1, 256, 56, 56]  |    512    |
|               Identity               | [-1, 256, 56, 56]  |     0     |
|     Add (Bottleneck + residual)      | [-1, 256, 56, 56]  |     0     |
|                 ReLU                 | [-1, 256, 56, 56]  |     0     |
|             **conv3_x**              |        ---         |    ---    |
|            2D convolution            | [-1, 128, 28, 28]  |  32,768   |
|            2D batch norm             | [-1, 128, 28, 28]  |    256    |
|                 ReLU                 | [-1, 128, 28, 28]  |     0     |
|            2D convolution            | [-1, 128, 28, 28]  |  147,456  |
|            2D batch norm             | [-1, 128, 28, 28]  |    256    |
|                 ReLU                 | [-1, 128, 28, 28]  |     0     |
|            2D convolution            | [-1, 512, 28, 28]  |  65,536   |
|            2D batch norm             | [-1, 512, 28, 28]  |   1,024   |
| 2D convolution (Residual projection) | [-1, 512, 28, 28]  |  131,072  |
| 2D batch norm (Residual projection)  | [-1, 512, 28, 28]  |   1,024   |
|     Add (Bottleneck + residual)      | [-1, 512, 28, 28]  |     0     |
|                 ReLU                 | [-1, 512, 28, 28]  |     0     |
|            2D convolution            | [-1, 128, 28, 28]  |  65,536   |
|            2D batch norm             | [-1, 128, 28, 28]  |    256    |
|                 ReLU                 | [-1, 128, 28, 28]  |     0     |
|            2D convolution            | [-1, 128, 28, 28]  |  147,456  |
|            2D batch norm             | [-1, 128, 28, 28]  |    256    |
|                 ReLU                 | [-1, 128, 28, 28]  |     0     |
|            2D convolution            | [-1, 512, 28, 28]  |  65,536   |
|            2D batch norm             | [-1, 512, 28, 28]  |   1,024   |
|               Identity               | [-1, 512, 28, 28]  |     0     |
|                 ReLU                 | [-1, 512, 28, 28]  |     0     |
|     Add (Bottleneck + residual)      | [-1, 512, 28, 28]  |     0     |
|                 ReLU                 | [-1, 512, 28, 28]  |     0     |
|            2D convolution            | [-1, 128, 28, 28]  |  65,536   |
|            2D batch norm             | [-1, 128, 28, 28]  |    256    |
|                 ReLU                 | [-1, 128, 28, 28]  |     0     |
|            2D convolution            | [-1, 128, 28, 28]  |  147,456  |
|            2D batch norm             | [-1, 128, 28, 28]  |    256    |
|                 ReLU                 | [-1, 128, 28, 28]  |     0     |
|            2D convolution            | [-1, 512, 28, 28]  |  65,536   |
|            2D batch norm             | [-1, 512, 28, 28]  |   1,024   |
|               Identity               | [-1, 512, 28, 28]  |     0     |
|                 ReLU                 | [-1, 512, 28, 28]  |     0     |
|     Add (Bottleneck + residual)      | [-1, 512, 28, 28]  |     0     |
|                 ReLU                 | [-1, 512, 28, 28]  |     0     |
|            2D convolution            | [-1, 128, 28, 28]  |  65,536   |
|            2D batch norm             | [-1, 128, 28, 28]  |    256    |
|                 ReLU                 | [-1, 128, 28, 28]  |     0     |
|            2D convolution            | [-1, 128, 28, 28]  |  147,456  |
|            2D batch norm             | [-1, 128, 28, 28]  |    256    |
|                 ReLU                 | [-1, 128, 28, 28]  |     0     |
|            2D convolution            | [-1, 512, 28, 28]  |  65,536   |
|            2D batch norm             | [-1, 512, 28, 28]  |   1,024   |
|               Identity               | [-1, 512, 28, 28]  |     0     |
|                 ReLU                 | [-1, 512, 28, 28]  |     0     |
|     Add (Bottleneck + residual)      | [-1, 512, 28, 28]  |     0     |
|                 ReLU                 | [-1, 512, 28, 28]  |     0     |
|             **conv4_x**              |        ---         |    ---    |
|            2D convolution            | [-1, 256, 14, 14]  |  131,072  |
|            2D batch norm             | [-1, 256, 14, 14]  |    512    |
|                 ReLU                 | [-1, 256, 14, 14]  |     0     |
|            2D convolution            | [-1, 256, 14, 14]  |  589,824  |
|            2D batch norm             | [-1, 256, 14, 14]  |    512    |
|                 ReLU                 | [-1, 256, 14, 14]  |     0     |
|            2D convolution            | [-1, 1024, 14, 14] |  262,144  |
|            2D batch norm             | [-1, 1024, 14, 14] |   2,048   |
| 2D convolution (Residual projection) | [-1, 1024, 14, 14] |  524,288  |
| 2D batch norm (Residual projection)  | [-1, 1024, 14, 14] |   2,048   |
|                 ReLU                 | [-1, 1024, 14, 14] |     0     |
|     Add (Bottleneck + residual)      | [-1, 1024, 14, 14] |     0     |
|                 ReLU                 | [-1, 1024, 14, 14] |     0     |
|            2D convolution            | [-1, 256, 14, 14]  |  262,144  |
|            2D batch norm             | [-1, 256, 14, 14]  |    512    |
|                 ReLU                 | [-1, 256, 14, 14]  |     0     |
|            2D convolution            | [-1, 256, 14, 14]  |  589,824  |
|            2D batch norm             | [-1, 256, 14, 14]  |    512    |
|                 ReLU                 | [-1, 256, 14, 14]  |     0     |
|            2D convolution            | [-1, 1024, 14, 14] |  262,144  |
|            2D batch norm             | [-1, 1024, 14, 14] |   2,048   |
|               Identity               | [-1, 1024, 14, 14] |     0     |
|                 ReLU                 | [-1, 1024, 14, 14] |     0     |
|     Add (Bottleneck + residual)      | [-1, 1024, 14, 14] |     0     |
|                 ReLU                 | [-1, 1024, 14, 14] |     0     |
|            2D convolution            | [-1, 256, 14, 14]  |  262,144  |
|            2D batch norm             | [-1, 256, 14, 14]  |    512    |
|                 ReLU                 | [-1, 256, 14, 14]  |     0     |
|            2D convolution            | [-1, 256, 14, 14]  |  589,824  |
|            2D batch norm             | [-1, 256, 14, 14]  |    512    |
|                 ReLU                 | [-1, 256, 14, 14]  |     0     |
|            2D convolution            | [-1, 1024, 14, 14] |  262,144  |
|            2D batch norm             | [-1, 1024, 14, 14] |   2,048   |
|               Identity               | [-1, 1024, 14, 14] |     0     |
|                 ReLU                 | [-1, 1024, 14, 14] |     0     |
|     Add (Bottleneck + residual)      | [-1, 1024, 14, 14] |     0     |
|                 ReLU                 | [-1, 1024, 14, 14] |     0     |
|            2D convolution            | [-1, 256, 14, 14]  |  262,144  |
|            2D batch norm             | [-1, 256, 14, 14]  |    512    |
|                 ReLU                 | [-1, 256, 14, 14]  |     0     |
|            2D convolution            | [-1, 256, 14, 14]  |  589,824  |
|            2D batch norm             | [-1, 256, 14, 14]  |    512    |
|                 ReLU                 | [-1, 256, 14, 14]  |     0     |
|            2D convolution            | [-1, 1024, 14, 14] |  262,144  |
|            2D batch norm             | [-1, 1024, 14, 14] |   2,048   |
|               Identity               | [-1, 1024, 14, 14] |     0     |
|                 ReLU                 | [-1, 1024, 14, 14] |     0     |
|     Add (Bottleneck + residual)      | [-1, 1024, 14, 14] |     0     |
|                 ReLU                 | [-1, 1024, 14, 14] |     0     |
|            2D convolution            | [-1, 256, 14, 14]  |  262,144  |
|            2D batch norm             | [-1, 256, 14, 14]  |    512    |
|                 ReLU                 | [-1, 256, 14, 14]  |     0     |
|            2D convolution            | [-1, 256, 14, 14]  |  589,824  |
|            2D batch norm             | [-1, 256, 14, 14]  |    512    |
|                 ReLU                 | [-1, 256, 14, 14]  |     0     |
|            2D convolution            | [-1, 1024, 14, 14] |  262,144  |
|            2D batch norm             | [-1, 1024, 14, 14] |   2,048   |
|               Identity               | [-1, 1024, 14, 14] |     0     |
|                 ReLU                 | [-1, 1024, 14, 14] |     0     |
|     Add (Bottleneck + residual)      | [-1, 1024, 14, 14] |     0     |
|                 ReLU                 | [-1, 1024, 14, 14] |     0     |
|            2D convolution            | [-1, 256, 14, 14]  |  262,144  |
|            2D batch norm             | [-1, 256, 14, 14]  |    512    |
|                 ReLU                 | [-1, 256, 14, 14]  |     0     |
|            2D convolution            | [-1, 256, 14, 14]  |  589,824  |
|            2D batch norm             | [-1, 256, 14, 14]  |    512    |
|                 ReLU                 | [-1, 256, 14, 14]  |     0     |
|            2D convolution            | [-1, 1024, 14, 14] |  262,144  |
|            2D batch norm             | [-1, 1024, 14, 14] |   2,048   |
|               Identity               | [-1, 1024, 14, 14] |     0     |
|                 ReLU                 | [-1, 1024, 14, 14] |     0     |
|     Add (Bottleneck + residual)      | [-1, 1024, 14, 14] |     0     |
|                 ReLU                 | [-1, 1024, 14, 14] |     0     |
|             **conv5_x**              |        ---         |    ---    |
|            2D convolution            |  [-1, 512, 7, 7]   |  524,288  |
|            2D batch norm             |  [-1, 512, 7, 7]   |   1,024   |
|                 ReLU                 |  [-1, 512, 7, 7]   |     0     |
|            2D convolution            |  [-1, 512, 7, 7]   | 2,359,296 |
|            2D batch norm             |  [-1, 512, 7, 7]   |   1,024   |
|                 ReLU                 |  [-1, 512, 7, 7]   |     0     |
|            2D convolution            |  [-1, 2048, 7, 7]  | 1,048,576 |
|            2D batch norm             |  [-1, 2048, 7, 7]  |   4,096   |
| 2D convolution (Residual projection) |  [-1, 2048, 7, 7]  | 2,097,152 |
| 2D batch norm (Residual projection)  |  [-1, 2048, 7, 7]  |   4,096   |
|                 ReLU                 |  [-1, 2048, 7, 7]  |     0     |
|     Add (Bottleneck + residual)      |  [-1, 2048, 7, 7]  |     0     |
|                 ReLU                 |  [-1, 2048, 7, 7]  |     0     |
|            2D convolution            |  [-1, 512, 7, 7]   | 1,048,576 |
|            2D batch norm             |  [-1, 512, 7, 7]   |   1,024   |
|                 ReLU                 |  [-1, 512, 7, 7]   |     0     |
|            2D convolution            |  [-1, 512, 7, 7]   | 2,359,296 |
|            2D batch norm             |  [-1, 512, 7, 7]   |   1,024   |
|                 ReLU                 |  [-1, 512, 7, 7]   |     0     |
|            2D convolution            |  [-1, 2048, 7, 7]  | 1,048,576 |
|            2D batch norm             |  [-1, 2048, 7, 7]  |   4,096   |
|               Identity               |  [-1, 2048, 7, 7]  |     0     |
|                 ReLU                 |  [-1, 2048, 7, 7]  |     0     |
|     Add (Bottleneck + residual)      |  [-1, 2048, 7, 7]  |     0     |
|                 ReLU                 |  [-1, 2048, 7, 7]  |     0     |
|            2D convolution            |  [-1, 512, 7, 7]   | 1,048,576 |
|            2D batch norm             |  [-1, 512, 7, 7]   |   1,024   |
|                 ReLU                 |  [-1, 512, 7, 7]   |     0     |
|            2D convolution            |  [-1, 512, 7, 7]   | 2,359,296 |
|            2D batch norm             |  [-1, 512, 7, 7]   |   1,024   |
|                 ReLU                 |  [-1, 512, 7, 7]   |     0     |
|            2D convolution            |  [-1, 2048, 7, 7]  | 1,048,576 |
|            2D batch norm             |  [-1, 2048, 7, 7]  |   4,096   |
|               Identity               |  [-1, 2048, 7, 7]  |     0     |
|                 ReLU                 |  [-1, 2048, 7, 7]  |     0     |
|     Add (Bottleneck + residual)      |  [-1, 2048, 7, 7]  |     0     |
|                 ReLU                 |  [-1, 2048, 7, 7]  |     0     |
|          2D average pooling          |  [-1, 2048, 1, 1]  |     0     |
|               Flatten                |     [-1, 2048]     |     0     |
|                Linear                |      [-1, 10]      |  20,490   |

|                  |            |
| ---------------- | ---------- |
| Total params     | 23,528,522 |
| Trainable params | 23,528,522 |
| Non-trainable    | params: 0  |

### Results

Training over 10 epochs with a learning rate of 1e-4 and batch size of 64.

|                     Loss                      |                       Accuracy                        |
| :-------------------------------------------: | :---------------------------------------------------: |
| ![MNIST loss graph](Resources/mnist_loss.png) | ![MNIST accuracy graph](Resources/mnist_accuracy.png) |

|              | Training | Validation | Testing |
| :----------: | :------: | :--------: | :-----: |
|     Loss     |  0.0114  |   0.0317   | 0.0268  |
| Accuracy (%) |  99.65   |   99.09    |  99.23  |

| Class | Training Precision | Validation Precision | Testing Precision | Training Recall | Validation Recall | Testing Recall | Training F1 Score | Validation F1 Score | Testing F1 Score |
| :---: | :----------------: | :------------------: | :---------------: | :-------------: | :---------------: | :------------: | :---------------: | :-----------------: | :--------------: |
|   0   |       0.9986       |        0.9949        |      0.9939       |     0.9981      |      0.9937       |     0.9980     |      0.9983       |       0.9943        |      0.9959      |
|   1   |       0.9974       |        0.9951        |      0.9947       |     0.9981      |      0.9971       |     0.9982     |      0.9978       |       0.9961        |      0.9965      |
|   2   |       0.9960       |        0.9932        |      0.9952       |     0.9952      |      0.9937       |     0.9971     |      0.9956       |       0.9934        |      0.9961      |
|   3   |       0.9977       |        0.9838        |      0.9777       |     0.9970      |      0.9984       |     0.9980     |      0.9973       |       0.9910        |      0.9878      |
|   4   |       0.9968       |        0.9837        |      0.9869       |     0.9958      |      0.9949       |     0.9939     |      0.9963       |       0.9893        |      0.9904      |
|   5   |       0.9960       |        0.9939        |      0.9966       |     0.9963      |      0.9789       |     0.9787     |      0.9961       |       0.9863        |      0.9876      |
|   6   |       0.9978       |        0.9893        |      0.9948       |     0.9976      |      0.9977       |     0.9948     |      0.9977       |       0.9935        |      0.9948      |
|   7   |       0.9952       |        0.9943        |      0.9971       |     0.9972      |      0.9876       |     0.9883     |      0.9962       |       0.9910        |      0.9927      |
|   8   |       0.9951       |        0.9947        |      0.9969       |     0.9949      |      0.9844       |     0.9908     |      0.9950       |       0.9895        |      0.9938      |
|   9   |       0.9940       |        0.9864        |      0.9900       |     0.9943      |      0.9813       |     0.9832     |      0.9941       |       0.9838        |      0.9866      |

Similar to all the preceding models, ResNet performs with close the human accuracy. It performs ever so slightly better than VGG does while containing ~18% of its parameters.

## References

Research paper: https://arxiv.org/pdf/1512.03385.pdf
