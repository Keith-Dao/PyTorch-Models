# Inception-V3

Implementation of Inception-V3 following the descriptions provided in the research paper.

Note that the research paper does not provide exact channel counts for the internals of the inception blocks, so the implementation may not be an exact replica.

## MNIST

### Model summary

|     Layer (type)     |    Output Shape    |  Param #  |
| :------------------: | :----------------: | :-------: |
|    2D convolution    | [-1, 32, 149, 149] |    864    |
|    2D batch norm     | [-1, 32, 149, 149] |    64     |
|         ReLU         | [-1, 32, 149, 149] |     0     |
|    2D convolution    | [-1, 32, 147, 147] |   9,216   |
|    2D batch norm     | [-1, 32, 147, 147] |    64     |
|         ReLU         | [-1, 32, 147, 147] |     0     |
|    2D convolution    | [-1, 64, 147, 147] |  18,432   |
|    2D batch norm     | [-1, 64, 147, 147] |    128    |
|         ReLU         | [-1, 64, 147, 147] |     0     |
|    2D max pooling    |  [-1, 64, 73, 73]  |     0     |
|    2D convolution    |  [-1, 80, 71, 71]  |  46,080   |
|    2D batch norm     |  [-1, 80, 71, 71]  |    160    |
|         ReLU         |  [-1, 80, 71, 71]  |     0     |
|    2D convolution    | [-1, 192, 35, 35]  |  138,240  |
|    2D batch norm     | [-1, 192, 35, 35]  |    384    |
|         ReLU         | [-1, 192, 35, 35]  |     0     |
| **inception fig 5a** |        ---         |    ---    |
|    2D convolution    |  [-1, 64, 35, 35]  |  12,288   |
|    2D batch norm     |  [-1, 64, 35, 35]  |    128    |
|         ReLU         |  [-1, 64, 35, 35]  |     0     |
|    2D convolution    |  [-1, 64, 35, 35]  |  36,864   |
|    2D batch norm     |  [-1, 64, 35, 35]  |    128    |
|         ReLU         |  [-1, 64, 35, 35]  |     0     |
|    2D convolution    |  [-1, 64, 35, 35]  |  36,864   |
|    2D batch norm     |  [-1, 64, 35, 35]  |    128    |
|         ReLU         |  [-1, 64, 35, 35]  |     0     |
|    2D convolution    |  [-1, 64, 35, 35]  |  12,288   |
|    2D batch norm     |  [-1, 64, 35, 35]  |    128    |
|         ReLU         |  [-1, 64, 35, 35]  |     0     |
|    2D convolution    |  [-1, 64, 35, 35]  |  36,864   |
|    2D batch norm     |  [-1, 64, 35, 35]  |    128    |
|         ReLU         |  [-1, 64, 35, 35]  |     0     |
|    2D max pooling    | [-1, 192, 35, 35]  |     0     |
|    2D convolution    |  [-1, 96, 35, 35]  |  18,432   |
|    2D batch norm     |  [-1, 96, 35, 35]  |    192    |
|         ReLU         |  [-1, 96, 35, 35]  |     0     |
|    2D convolution    |  [-1, 32, 35, 35]  |   6,144   |
|    2D batch norm     |  [-1, 32, 35, 35]  |    64     |
|         ReLU         |  [-1, 32, 35, 35]  |     0     |
| **inception fig 5b** |        ---         |    ---    |
|    2D convolution    |  [-1, 64, 35, 35]  |  16,384   |
|    2D batch norm     |  [-1, 64, 35, 35]  |    128    |
|         ReLU         |  [-1, 64, 35, 35]  |     0     |
|    2D convolution    |  [-1, 64, 35, 35]  |  36,864   |
|    2D batch norm     |  [-1, 64, 35, 35]  |    128    |
|         ReLU         |  [-1, 64, 35, 35]  |     0     |
|    2D convolution    |  [-1, 64, 35, 35]  |  36,864   |
|    2D batch norm     |  [-1, 64, 35, 35]  |    128    |
|         ReLU         |  [-1, 64, 35, 35]  |     0     |
|    2D convolution    |  [-1, 64, 35, 35]  |  16,384   |
|    2D batch norm     |  [-1, 64, 35, 35]  |    128    |
|         ReLU         |  [-1, 64, 35, 35]  |     0     |
|    2D convolution    |  [-1, 64, 35, 35]  |  36,864   |
|    2D batch norm     |  [-1, 64, 35, 35]  |    128    |
|         ReLU         |  [-1, 64, 35, 35]  |     0     |
|    2D max pooling    | [-1, 256, 35, 35]  |     0     |
|    2D convolution    |  [-1, 96, 35, 35]  |  24,576   |
|    2D batch norm     |  [-1, 96, 35, 35]  |    192    |
|         ReLU         |  [-1, 96, 35, 35]  |     0     |
|    2D convolution    |  [-1, 64, 35, 35]  |  16,384   |
|    2D batch norm     |  [-1, 64, 35, 35]  |    128    |
|         ReLU         |  [-1, 64, 35, 35]  |     0     |
| **inception fig 5c** |        ---         |    ---    |
|    2D convolution    |  [-1, 64, 35, 35]  |  18,432   |
|    2D batch norm     |  [-1, 64, 35, 35]  |    128    |
|         ReLU         |  [-1, 64, 35, 35]  |     0     |
|    2D convolution    |  [-1, 64, 35, 35]  |  36,864   |
|    2D batch norm     |  [-1, 64, 35, 35]  |    128    |
|         ReLU         |  [-1, 64, 35, 35]  |     0     |
|    2D convolution    |  [-1, 64, 35, 35]  |  36,864   |
|    2D batch norm     |  [-1, 64, 35, 35]  |    128    |
|         ReLU         |  [-1, 64, 35, 35]  |     0     |
|    2D convolution    |  [-1, 64, 35, 35]  |  18,432   |
|    2D batch norm     |  [-1, 64, 35, 35]  |    128    |
|         ReLU         |  [-1, 64, 35, 35]  |     0     |
|    2D convolution    |  [-1, 64, 35, 35]  |  36,864   |
|    2D batch norm     |  [-1, 64, 35, 35]  |    128    |
|         ReLU         |  [-1, 64, 35, 35]  |     0     |
|    2D max pooling    | [-1, 288, 35, 35]  |     0     |
|    2D convolution    |  [-1, 96, 35, 35]  |  27,648   |
|    2D batch norm     |  [-1, 96, 35, 35]  |    192    |
|         ReLU         |  [-1, 96, 35, 35]  |     0     |
|    2D convolution    |  [-1, 64, 35, 35]  |  18,432   |
|    2D batch norm     |  [-1, 64, 35, 35]  |    128    |
|         ReLU         |  [-1, 64, 35, 35]  |     0     |
|  **grid reduction**  |        ---         |    ---    |
|    2D convolution    | [-1, 384, 35, 35]  |  110,592  |
|    2D batch norm     | [-1, 384, 35, 35]  |    768    |
|         ReLU         | [-1, 384, 35, 35]  |     0     |
|    2D convolution    | [-1, 384, 35, 35]  | 1,327,104 |
|    2D batch norm     | [-1, 384, 35, 35]  |    768    |
|         ReLU         | [-1, 384, 35, 35]  |     0     |
|    2D convolution    | [-1, 384, 17, 17]  | 1,327,104 |
|    2D batch norm     | [-1, 384, 17, 17]  |    768    |
|         ReLU         | [-1, 384, 17, 17]  |     0     |
|    2D convolution    |  [-1, 96, 35, 35]  |  27,648   |
|    2D batch norm     |  [-1, 96, 35, 35]  |    192    |
|         ReLU         |  [-1, 96, 35, 35]  |     0     |
|    2D convolution    |  [-1, 96, 17, 17]  |  82,944   |
|    2D batch norm     |  [-1, 96, 17, 17]  |    192    |
|         ReLU         |  [-1, 96, 17, 17]  |     0     |
|    2D max pooling    | [-1, 288, 17, 17]  |     0     |
| **inception fig 6a** |        ---         |    ---    |
|    2D convolution    | [-1, 128, 17, 17]  |  98,304   |
|    2D batch norm     | [-1, 128, 17, 17]  |    256    |
|         ReLU         | [-1, 128, 17, 17]  |     0     |
|    2D convolution    | [-1, 128, 17, 17]  |  114,688  |
|    2D batch norm     | [-1, 128, 17, 17]  |    256    |
|         ReLU         | [-1, 128, 17, 17]  |     0     |
|    2D convolution    | [-1, 128, 17, 17]  |  114,688  |
|    2D batch norm     | [-1, 128, 17, 17]  |    256    |
|         ReLU         | [-1, 128, 17, 17]  |     0     |
|    2D convolution    | [-1, 128, 17, 17]  |  114,688  |
|    2D batch norm     | [-1, 128, 17, 17]  |    256    |
|         ReLU         | [-1, 128, 17, 17]  |     0     |
|    2D convolution    | [-1, 192, 17, 17]  |  172,032  |
|    2D batch norm     | [-1, 192, 17, 17]  |    384    |
|         ReLU         | [-1, 192, 17, 17]  |     0     |
|    2D convolution    | [-1, 128, 17, 17]  |  98,304   |
|    2D batch norm     | [-1, 128, 17, 17]  |    256    |
|         ReLU         | [-1, 128, 17, 17]  |     0     |
|    2D convolution    | [-1, 128, 17, 17]  |  114,688  |
|    2D batch norm     | [-1, 128, 17, 17]  |    256    |
|         ReLU         | [-1, 128, 17, 17]  |     0     |
|    2D convolution    | [-1, 192, 17, 17]  |  172,032  |
|    2D batch norm     | [-1, 192, 17, 17]  |    384    |
|         ReLU         | [-1, 192, 17, 17]  |     0     |
|    2D max pooling    | [-1, 768, 17, 17]  |     0     |
|    2D convolution    | [-1, 192, 17, 17]  |  147,456  |
|    2D batch norm     | [-1, 192, 17, 17]  |    384    |
|         ReLU         | [-1, 192, 17, 17]  |     0     |
|    2D convolution    | [-1, 192, 17, 17]  |  147,456  |
|    2D batch norm     | [-1, 192, 17, 17]  |    384    |
|         ReLU         | [-1, 192, 17, 17]  |     0     |
| **inception fig 6b** |        ---         |    ---    |
|    2D convolution    | [-1, 144, 17, 17]  |  110,592  |
|    2D batch norm     | [-1, 144, 17, 17]  |    288    |
|         ReLU         | [-1, 144, 17, 17]  |     0     |
|    2D convolution    | [-1, 144, 17, 17]  |  145,152  |
|    2D batch norm     | [-1, 144, 17, 17]  |    288    |
|         ReLU         | [-1, 144, 17, 17]  |     0     |
|    2D convolution    | [-1, 144, 17, 17]  |  145,152  |
|    2D batch norm     | [-1, 144, 17, 17]  |    288    |
|         ReLU         | [-1, 144, 17, 17]  |     0     |
|    2D convolution    | [-1, 144, 17, 17]  |  145,152  |
|    2D batch norm     | [-1, 144, 17, 17]  |    288    |
|         ReLU         | [-1, 144, 17, 17]  |     0     |
|    2D convolution    | [-1, 192, 17, 17]  |  193,536  |
|    2D batch norm     | [-1, 192, 17, 17]  |    384    |
|         ReLU         | [-1, 192, 17, 17]  |     0     |
|    2D convolution    | [-1, 144, 17, 17]  |  110,592  |
|    2D batch norm     | [-1, 144, 17, 17]  |    288    |
|         ReLU         | [-1, 144, 17, 17]  |     0     |
|    2D convolution    | [-1, 144, 17, 17]  |  145,152  |
|    2D batch norm     | [-1, 144, 17, 17]  |    288    |
|         ReLU         | [-1, 144, 17, 17]  |     0     |
|    2D convolution    | [-1, 192, 17, 17]  |  193,536  |
|    2D batch norm     | [-1, 192, 17, 17]  |    384    |
|         ReLU         | [-1, 192, 17, 17]  |     0     |
|    2D max pooling    | [-1, 768, 17, 17]  |     0     |
|    2D convolution    | [-1, 192, 17, 17]  |  147,456  |
|    2D batch norm     | [-1, 192, 17, 17]  |    384    |
|         ReLU         | [-1, 192, 17, 17]  |     0     |
|    2D convolution    | [-1, 192, 17, 17]  |  147,456  |
|    2D batch norm     | [-1, 192, 17, 17]  |    384    |
|         ReLU         | [-1, 192, 17, 17]  |     0     |
| **inception fig 6c** |        ---         |    ---    |
|    2D convolution    | [-1, 160, 17, 17]  |  122,880  |
|    2D batch norm     | [-1, 160, 17, 17]  |    320    |
|         ReLU         | [-1, 160, 17, 17]  |     0     |
|    2D convolution    | [-1, 160, 17, 17]  |  179,200  |
|    2D batch norm     | [-1, 160, 17, 17]  |    320    |
|         ReLU         | [-1, 160, 17, 17]  |     0     |
|    2D convolution    | [-1, 160, 17, 17]  |  179,200  |
|    2D batch norm     | [-1, 160, 17, 17]  |    320    |
|         ReLU         | [-1, 160, 17, 17]  |     0     |
|    2D convolution    | [-1, 160, 17, 17]  |  179,200  |
|    2D batch norm     | [-1, 160, 17, 17]  |    320    |
|         ReLU         | [-1, 160, 17, 17]  |     0     |
|    2D convolution    | [-1, 192, 17, 17]  |  215,040  |
|    2D batch norm     | [-1, 192, 17, 17]  |    384    |
|         ReLU         | [-1, 192, 17, 17]  |     0     |
|    2D convolution    | [-1, 160, 17, 17]  |  122,880  |
|    2D batch norm     | [-1, 160, 17, 17]  |    320    |
|         ReLU         | [-1, 160, 17, 17]  |     0     |
|    2D convolution    | [-1, 160, 17, 17]  |  179,200  |
|    2D batch norm     | [-1, 160, 17, 17]  |    320    |
|         ReLU         | [-1, 160, 17, 17]  |     0     |
|    2D convolution    | [-1, 192, 17, 17]  |  215,040  |
|    2D batch norm     | [-1, 192, 17, 17]  |    384    |
|         ReLU         | [-1, 192, 17, 17]  |     0     |
|    2D max pooling    | [-1, 768, 17, 17]  |     0     |
|    2D convolution    | [-1, 192, 17, 17]  |  147,456  |
|    2D batch norm     | [-1, 192, 17, 17]  |    384    |
|         ReLU         | [-1, 192, 17, 17]  |     0     |
|    2D convolution    | [-1, 192, 17, 17]  |  147,456  |
|    2D batch norm     | [-1, 192, 17, 17]  |    384    |
|         ReLU         | [-1, 192, 17, 17]  |     0     |
| **inception fig 6c** |        ---         |    ---    |
|    2D convolution    | [-1, 176, 17, 17]  |  135,168  |
|    2D batch norm     | [-1, 176, 17, 17]  |    352    |
|         ReLU         | [-1, 176, 17, 17]  |     0     |
|    2D convolution    | [-1, 176, 17, 17]  |  216,832  |
|    2D batch norm     | [-1, 176, 17, 17]  |    352    |
|         ReLU         | [-1, 176, 17, 17]  |     0     |
|    2D convolution    | [-1, 176, 17, 17]  |  216,832  |
|    2D batch norm     | [-1, 176, 17, 17]  |    352    |
|         ReLU         | [-1, 176, 17, 17]  |     0     |
|    2D convolution    | [-1, 176, 17, 17]  |  216,832  |
|    2D batch norm     | [-1, 176, 17, 17]  |    352    |
|         ReLU         | [-1, 176, 17, 17]  |     0     |
|    2D convolution    | [-1, 192, 17, 17]  |  236,544  |
|    2D batch norm     | [-1, 192, 17, 17]  |    384    |
|         ReLU         | [-1, 192, 17, 17]  |     0     |
|    2D convolution    | [-1, 176, 17, 17]  |  135,168  |
|    2D batch norm     | [-1, 176, 17, 17]  |    352    |
|         ReLU         | [-1, 176, 17, 17]  |     0     |
|    2D convolution    | [-1, 176, 17, 17]  |  216,832  |
|    2D batch norm     | [-1, 176, 17, 17]  |    352    |
|         ReLU         | [-1, 176, 17, 17]  |     0     |
|    2D convolution    | [-1, 192, 17, 17]  |  236,544  |
|    2D batch norm     | [-1, 192, 17, 17]  |    384    |
|         ReLU         | [-1, 192, 17, 17]  |     0     |
|    2D max pooling    | [-1, 768, 17, 17]  |     0     |
|    2D convolution    | [-1, 192, 17, 17]  |  147,456  |
|    2D batch norm     | [-1, 192, 17, 17]  |    384    |
|         ReLU         | [-1, 192, 17, 17]  |     0     |
|    2D convolution    | [-1, 192, 17, 17]  |  147,456  |
|    2D batch norm     | [-1, 192, 17, 17]  |    384    |
|         ReLU         | [-1, 192, 17, 17]  |     0     |
| **inception fig 6d** |        ---         |    ---    |
|    2D convolution    | [-1, 192, 17, 17]  |  147,456  |
|    2D batch norm     | [-1, 192, 17, 17]  |    384    |
|         ReLU         | [-1, 192, 17, 17]  |     0     |
|    2D convolution    | [-1, 192, 17, 17]  |  258,048  |
|    2D batch norm     | [-1, 192, 17, 17]  |    384    |
|         ReLU         | [-1, 192, 17, 17]  |     0     |
|    2D convolution    | [-1, 192, 17, 17]  |  258,048  |
|    2D batch norm     | [-1, 192, 17, 17]  |    384    |
|         ReLU         | [-1, 192, 17, 17]  |     0     |
|    2D convolution    | [-1, 192, 17, 17]  |  258,048  |
|    2D batch norm     | [-1, 192, 17, 17]  |    384    |
|         ReLU         | [-1, 192, 17, 17]  |     0     |
|    2D convolution    | [-1, 192, 17, 17]  |  258,048  |
|    2D batch norm     | [-1, 192, 17, 17]  |    384    |
|         ReLU         | [-1, 192, 17, 17]  |     0     |
|    2D convolution    | [-1, 192, 17, 17]  |  147,456  |
|    2D batch norm     | [-1, 192, 17, 17]  |    384    |
|         ReLU         | [-1, 192, 17, 17]  |     0     |
|    2D convolution    | [-1, 192, 17, 17]  |  258,048  |
|    2D batch norm     | [-1, 192, 17, 17]  |    384    |
|         ReLU         | [-1, 192, 17, 17]  |     0     |
|    2D convolution    | [-1, 192, 17, 17]  |  258,048  |
|    2D batch norm     | [-1, 192, 17, 17]  |    384    |
|         ReLU         | [-1, 192, 17, 17]  |     0     |
|    2D max pooling    | [-1, 768, 17, 17]  |     0     |
|    2D convolution    | [-1, 192, 17, 17]  |  147,456  |
|    2D batch norm     | [-1, 192, 17, 17]  |    384    |
|         ReLU         | [-1, 192, 17, 17]  |     0     |
|    2D convolution    | [-1, 192, 17, 17]  |  147,456  |
|    2D batch norm     | [-1, 192, 17, 17]  |    384    |
|         ReLU         | [-1, 192, 17, 17]  |     0     |
|       **aux**        |        ---         |    ---    |
|    AvgPool2d-344     |  [-1, 768, 5, 5]   |     0     |
|    2D convolution    |  [-1, 128, 5, 5]   |  98,304   |
|    2D batch norm     |  [-1, 128, 5, 5]   |    256    |
|         ReLU         |  [-1, 128, 5, 5]   |     0     |
|       Flatten        |     [-1, 3200]     |     0     |
|        Linear        |     [-1, 1024]     | 3,277,824 |
|        Linear        |      [-1, 10]      |  10,250   |
|  **grid reduction**  |        ---         |    ---    |
|    2D convolution    | [-1, 384, 17, 17]  |  294,912  |
|    2D batch norm     | [-1, 384, 17, 17]  |    768    |
|         ReLU         | [-1, 384, 17, 17]  |     0     |
|    2D convolution    | [-1, 384, 17, 17]  | 1,327,104 |
|    2D batch norm     | [-1, 384, 17, 17]  |    768    |
|         ReLU         | [-1, 384, 17, 17]  |     0     |
|    2D convolution    |  [-1, 384, 8, 8]   | 1,327,104 |
|    2D batch norm     |  [-1, 384, 8, 8]   |    768    |
|         ReLU         |  [-1, 384, 8, 8]   |     0     |
|    2D convolution    | [-1, 128, 17, 17]  |  98,304   |
|    2D batch norm     | [-1, 128, 17, 17]  |    256    |
|         ReLU         | [-1, 128, 17, 17]  |     0     |
|    2D convolution    |  [-1, 128, 8, 8]   |  147,456  |
|    2D batch norm     |  [-1, 128, 8, 8]   |    256    |
|         ReLU         |  [-1, 128, 8, 8]   |     0     |
|    2D max pooling    |  [-1, 768, 8, 8]   |     0     |
| **inception fig 7a** |        ---         |    ---    |
|    2D convolution    |  [-1, 208, 8, 8]   |  266,240  |
|    2D batch norm     |  [-1, 208, 8, 8]   |    416    |
|         ReLU         |  [-1, 208, 8, 8]   |     0     |
|    2D convolution    |  [-1, 208, 8, 8]   |  389,376  |
|    2D batch norm     |  [-1, 208, 8, 8]   |    416    |
|         ReLU         |  [-1, 208, 8, 8]   |     0     |
|    2D convolution    |  [-1, 208, 8, 8]   |  266,240  |
|    2D batch norm     |  [-1, 208, 8, 8]   |    416    |
|         ReLU         |  [-1, 208, 8, 8]   |     0     |
|    2D convolution    |  [-1, 208, 8, 8]   |  129,792  |
|    2D batch norm     |  [-1, 208, 8, 8]   |    416    |
|         ReLU         |  [-1, 208, 8, 8]   |     0     |
|    2D convolution    |  [-1, 208, 8, 8]   |  129,792  |
|    2D batch norm     |  [-1, 208, 8, 8]   |    416    |
|         ReLU         |  [-1, 208, 8, 8]   |     0     |
|    2D convolution    |  [-1, 208, 8, 8]   |  129,792  |
|    2D batch norm     |  [-1, 208, 8, 8]   |    416    |
|         ReLU         |  [-1, 208, 8, 8]   |     0     |
|    2D convolution    |  [-1, 208, 8, 8]   |  129,792  |
|    2D batch norm     |  [-1, 208, 8, 8]   |    416    |
|         ReLU         |  [-1, 208, 8, 8]   |     0     |
|    2D max pooling    |  [-1, 1280, 8, 8]  |     0     |
|    2D convolution    |  [-1, 416, 8, 8]   |  532,480  |
|    2D batch norm     |  [-1, 416, 8, 8]   |    832    |
|         ReLU         |  [-1, 416, 8, 8]   |     0     |
|    2D convolution    |  [-1, 416, 8, 8]   |  532,480  |
|    2D batch norm     |  [-1, 416, 8, 8]   |    832    |
|         ReLU         |  [-1, 416, 8, 8]   |     0     |
| **inception fig 7b** |        ---         |    ---    |
|    2D convolution    |  [-1, 256, 8, 8]   |  425,984  |
|    2D batch norm     |  [-1, 256, 8, 8]   |    512    |
|         ReLU         |  [-1, 256, 8, 8]   |     0     |
|    2D convolution    |  [-1, 256, 8, 8]   |  589,824  |
|    2D batch norm     |  [-1, 256, 8, 8]   |    512    |
|         ReLU         |  [-1, 256, 8, 8]   |     0     |
|    2D convolution    |  [-1, 256, 8, 8]   |  425,984  |
|    2D batch norm     |  [-1, 256, 8, 8]   |    512    |
|         ReLU         |  [-1, 256, 8, 8]   |     0     |
|    2D convolution    |  [-1, 256, 8, 8]   |  196,608  |
|    2D batch norm     |  [-1, 256, 8, 8]   |    512    |
|         ReLU         |  [-1, 256, 8, 8]   |     0     |
|    2D convolution    |  [-1, 256, 8, 8]   |  196,608  |
|    2D batch norm     |  [-1, 256, 8, 8]   |    512    |
|         ReLU         |  [-1, 256, 8, 8]   |     0     |
|    2D convolution    |  [-1, 256, 8, 8]   |  196,608  |
|    2D batch norm     |  [-1, 256, 8, 8]   |    512    |
|         ReLU         |  [-1, 256, 8, 8]   |     0     |
|    2D convolution    |  [-1, 256, 8, 8]   |  196,608  |
|    2D batch norm     |  [-1, 256, 8, 8]   |    512    |
|         ReLU         |  [-1, 256, 8, 8]   |     0     |
|    2D max pooling    |  [-1, 1664, 8, 8]  |     0     |
|    2D convolution    |  [-1, 512, 8, 8]   |  851,968  |
|    2D batch norm     |  [-1, 512, 8, 8]   |   1,024   |
|         ReLU         |  [-1, 512, 8, 8]   |     0     |
|    2D convolution    |  [-1, 512, 8, 8]   |  851,968  |
|    2D batch norm     |  [-1, 512, 8, 8]   |   1,024   |
|         ReLU         |  [-1, 512, 8, 8]   |     0     |
|    2D avg pooling    |  [-1, 2048, 1, 1]  |     0     |
|       Flatten        |     [-1, 2048]     |     0     |
|        Linear        |      [-1, 10]      |  20,490   |

|                                         |            |
| --------------------------------------- | ---------- |
| Total params                            | 25,212,020 |
| Trainable params                        | 25,212,020 |
| Total params excluding auxiliary params | 21,825,386 |
| Non-trainable params                    | 0          |

### Results

Trained over 10 epochs with a learning rate of 1e-4, batch size of 32, auxiliary loss weighting of 0.3, label smoothing factor of 1e-4 and weight decay of 0.01.

|                     Loss                      |                       Accuracy                        |
| :-------------------------------------------: | :---------------------------------------------------: |
| ![MNIST loss graph](Resources/mnist_loss.png) | ![MNIST accuracy graph](Resources/mnist_accuracy.png) |

|              | Training | Validation | Testing |
| :----------: | :------: | :--------: | :-----: |
|     Loss     |  0.0328  |   0.0276   | 0.0204  |
| Accuracy (%) |  99.32   |   99.31    |  99.52  |

| Class | Training Precision | Validation Precision | Testing Precision | Training Recall | Validation Recall | Testing Recall | Training F1 Score | Validation F1 Score | Testing F1 Score |
| :---: | :----------------: | :------------------: | :---------------: | :-------------: | :---------------: | :------------: | :---------------: | :-----------------: | :--------------: |
|   0   |       0.9953       |        0.9962        |      0.9969       |     0.9973      |      0.9962       |     0.9990     |      0.9963       |       0.9962        |      0.9980      |
|   1   |       0.9953       |        0.9936        |      0.9947       |     0.9956      |      0.9965       |     0.9947     |      0.9955       |       0.9950        |      0.9947      |
|   2   |       0.9933       |        0.9900        |      0.9923       |     0.9930      |      0.9939       |     0.9971     |      0.9931       |       0.9919        |      0.9947      |
|   3   |       0.9946       |        0.9983        |      0.9970       |     0.9928      |      0.9923       |     0.9960     |      0.9937       |       0.9953        |      0.9965      |
|   4   |       0.9910       |        0.9912        |      0.9959       |     0.9910      |      0.9912       |     0.9959     |      0.9910       |       0.9912        |      0.9959      |
|   5   |       0.9927       |        0.9943        |      0.9911       |     0.9925      |      0.9936       |     0.9966     |      0.9926       |       0.9940        |      0.9939      |
|   6   |       0.9934       |        0.9901        |      0.9927       |     0.9944      |      0.9972       |     0.9969     |      0.9939       |       0.9937        |      0.9948      |
|   7   |       0.9924       |        0.9961        |      0.9961       |     0.9944      |      0.9907       |     0.9883     |      0.9934       |       0.9934        |      0.9922      |
|   8   |       0.9934       |        0.9954        |      0.9990       |     0.9915      |      0.9858       |     0.9959     |      0.9924       |       0.9906        |      0.9974      |
|   9   |       0.9903       |        0.9854        |      0.9960       |     0.9893      |      0.9924       |     0.9921     |      0.9898       |       0.9889        |      0.9940      |

Compared to Inception-BN, Inception-V3 performs slightly better across the data sets but required more epochs to train. Despite containing almost twice as many parameters, the performance did not scale accordingly, which may be due to the MNIST dataset lacking significant complexity. From the loss and accuracy graphs, it is evident that the performance began regressing but was eventually able to recover. This recovery could be attributed to the use of label smoothing.

## References

Research paper: https://arxiv.org/pdf/1512.00567.pdf
