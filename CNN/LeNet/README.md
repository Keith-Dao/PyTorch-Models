# LeNet-5

This is a replication of the LeNet-5 model being as close to as described in the research paper and a reimplementation of the architecture using modern developments.

## Original

### Model Summary

| Layer (type)                     | Output Shape     | Param # |
| -------------------------------- | ---------------- | ------- |
| 2D Convolution layer (C1)        | [-1, 6, 28, 28]  | 156     |
| Tanh activation                  | [-1, 6, 28, 28]  | 0       |
| Subsampling layer (S2)           | [-1, 6, 14, 14]  | 12      |
| Tanh activation                  | [-1, 6, 14, 14]  | 0       |
| Sparse 2D Convolution layer (C3) | [-1, 16, 10, 10] | 1,516   |
| Tanh activation                  | [-1, 16, 10, 10] | 0       |
| Subsampling layer (S4)           | [-1, 16, 5, 5]   | 32      |
| Tanh activation                  | [-1, 16, 5, 5]   | 0       |
| 2D Convolution layer (C5)        | [-1, 120, 1, 1]  | 48,120  |
| Tanh activation                  | [-1, 120, 1, 1]  | 0       |
| Linear layer (F6)                | [-1, 84]         | 10,164  |
| RBF layer                        | [-1, 10]         | 0       |

|                      |        |
| -------------------- | ------ |
| Total params         | 60,000 |
| Trainable params     | 60,000 |
| Non-trainable params | 0      |

### Differences

The major change comes in the use of the Adam optimizer over the stochastic gradient descent optimizer. This change was necessary as training with SGD led to exploding gradients, thus causing the model to be unable to learn.

### Results

|                        Loss                         |                          Accuracy                           |
| :-------------------------------------------------: | :---------------------------------------------------------: |
| ![Original loss graph](Resources/original_loss.png) | ![Original accuracy graph](Resources/original_accuracy.png) |

|              | Training | Validation | Testing |
| :----------: | :------: | :--------: | :-----: |
|     Loss     |  353.03  |   352.91   | 337.71  |
| Accuracy (%) |  84.73   |   85.06    |  85.89  |

From the results, it can be seen that the model converges after training for 20 epochs with the specified learning rates and without any indications of over or under fitting.

## References

Research Paper: http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf
