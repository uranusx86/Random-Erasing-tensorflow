# Random-Erasing-tensorflow
A data argumentation implementation of random erasing using Tensorflow <br/>
WITHOUT numpy !

**Note: this side project is just for fun, the performance is equal to numpy version when using i5 CPU, but a bit lower than numpy version when using i7 CPU**

# Dependency
1. Tensorflow 1.10+

# Result
![](https://github.com/uranusx86/Random-Erasing-tensorflow/blob/master/data/random_erasing.jpg)

# Test Performance
```
python3 test_performance.py
```
Current version test @ CPU i5-3470 & GPU GTX1080ti <br/>
numpy: 0.1695 sec/image <br/>
TF: 0.1696 sec/image

# Ref
1. [Random Erasing Data Augmentation](http://arxiv.org/abs/1602.02830), Zhun Zhong, Liang Zheng, Guoliang Kang, Shaozi Li, Yi Yang
2. [Random Erasing (Author version)](https://github.com/zhunzhong07/Random-Erasing)
3. [Cutout Random Erasing (Keras version)](https://github.com/yu4u/cutout-random-erasing)