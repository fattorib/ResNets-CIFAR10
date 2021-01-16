# Deep Residual Learning for Image Recognition
PyTorch implementation of the CIFAR10 ResNets, based on *Deep Residual Learning for Image Recognition* ([He et al.](https://arxiv.org/abs/1512.03385))

# Short Paper Summary
The authors of this paper begin by noting that neural networks can benefit from increased depth. However, when the models get too deep, they note a degredation in the models performance. Interesting is that this is not caused by overfitting; both training and test error will saturate at higher values than more shallow network counterparts. See figure 1 from the paper:
<p align="center">
<img src="https://github.com/fattorib/ml_papers_code/blob/master/ResNets/fig1.PNG" width="450">
</p>

Consider a neural network that performs well on a dataset. If we were to add many identity layers before the output layer, our model would perform the exact same. Therefore, in theory a deeper network should perform as well, if not better than its more shallow counterpart. In the most extreme example, the model would learn many identity layers. The authors hypothesize that identity mappings may be difficult for a network to learn. 

To fix this, they introduce the idea of *deep residual learning*. Consider a group of stacked layers that take x as an input. If we let H(x) denote the optimal mapping, then we train the network to fit F(x) = H(x)-x instead. We do this through adding in the *residual connection* +x to our function F(x). Coming back to the above example, the authors hypothesize that it is easier to push all weights to 0 in F(x) than it is for the network to learn that F(x) = Id(x). Here is an example residual block: 
<p align="center">
<img src="https://github.com/fattorib/ml_papers_code/blob/master/ResNets/resnet.png" width="150">
</p>
In terms of connections, there are two main ways to do this:

A) Use the identity mapping. (This is the method we use for CIFAR10)

B) Use projections to match the output dimension.

## CIFAR10 Model
The authors demonstrate SOTA results on ImageNet and also include details on training models for CIFAR10 as well. These are the models we focus on. The architecture is as follows: 
| output map size | 32 x 32  | 16 x 16 | 8 x 8 |
|-----------------|----------|---------|-------|
| # layers        | 1+2n     | 2n      | 2n    |
| # filters       | 16       | 32      | 64    |

Shortcut connections are used to connect all pairs of 3x3 convolutions. For downsampling, we use convolutions with a stride of (2,2). 

# Models
We implement the ResNets corresponding to n={3,5,7,9}. The paper also introduces two larger ResNets, ResNet110 and ResNet1202 which correspond to n={18,200}. We skip these as the models are too large and time consuming for training on a single GTX 1070 (ResNet1202 requires 16gb(!) of VRAM). In addition, as mentioned in the paper, ResNet1202 actually has worse performance than ResNet32, most likely due to overparameterization. 

# Training Process
The models are trained on an augmented dataset (random 32x32 crop on a padded image and random horizontal flips applied). A batch size of 128 is used and the model is optimized using SGD with momentum with a starting learning rate of 0.01. Finally, we use Kaiming initialization for all model weights. 

The paper outlines dividing the learning rate by 10 at 32k and 48k iterations and ceasing training at 64k iterations. This corresponds to decreasing the learning rate at 90 and 135 epochs and ceasing training at 180 epochs. In our implementation this, training scheduling worked well for ResNet20 and ResNet32. The two larger ResNets benefitted from training for more epochs (200 instead of 180). 

# Results
The results we obtain almost identically match the results of the paper<sup>[1](#myfootnote1)</sup>. Improved performance could possibly be achieved by letting the models train on all of the data instead of 90% of it. Early stopping is not used during training so there is less point to keeping this holdout set. 


| Model    | Paper Top 1 Error % | My Implementation Top 1 Error % |
|----------|---------------------|---------------------------------|
| Resnet20 | 8.75                | 8.57                            |
| ResNet32 | 7.51                | 7.82                            |
| ResNet44 | 7.17                | 7.20                            |
| ResNet56 | 6.97                | 7.50                            |

# References:
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recog-nition.  InProceedings of the IEEE conference on computer vision and pattern recognition, pp.770–778, 2016.

<a name="myfootnote1">1</a>: ResNet56 slightly underperformed. Allowing it to train for more epochs would most likely fix this. Unfortunately, this model quite a long time to train even on better hardware(Tesla V100) and I only had limited access to it. 
