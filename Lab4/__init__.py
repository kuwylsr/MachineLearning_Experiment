'''
PCA.py文件 读取手动生成的三维高斯分布，对三维高斯分布数据进行PCA算法的主成分提取
PCA_mnist.py文件 读取特定的数字训练样本，对手写体数字数据进行PCA算法的主成分提取
ThreeDGaussianData.py文件 是手动生成的三维高斯分布数据，生成之后并写入excel表中

源文件中共存在7个文件：
    MNIST_test.txt 网站上下载的用于测试的手写体数字数据
    MNIST_train.txt 网站上下载的用于训练的手写体数字数据
    MultiGaussianData.xls 手动生成的三维高斯分布的数据
    test2.txt 从数据集中筛选出手写体数字为2的数字的（用于训练）文本
    test4.txt 从数据集中筛选出手写体数字为4的数字的（用于训练）文本
    test5.txt 从数据集中筛选出手写体数字为5的数字的（用于训练）文本
    test8.txt 从数据集中筛选出手写体数字为8的数字的（用于训练）文本
'''

__author__ = 'LiSirui 1160300610'
