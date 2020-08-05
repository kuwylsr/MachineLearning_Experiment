import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import matplotlib as mpl
import numpy as np
import random
import math


def ProductGN(n):
    #随机在[0,1]生成n个服从均匀分布的X点
    D = np.linspace(0,1,n)
    n_train = int(0.7*n)
    n_validate = int(0.2*n)
    n_test = int(0.1*n)
    #生成训练集合（Training Set）
    X_train = random.sample(list(D),n_train)
    Y_train = np.sin(2*np.pi*np.array(X_train)) #计算出响应的sin（）函数值
    #生成验证集合（Validation Set）
    X_validate = random.sample(list(D),n_validate)
    Y_validate = np.sin(2*np.pi*np.array(X_validate)) #计算出响应的sin（）函数值
    #生成测试集合（Test Set）
    X_test = random.sample(list(D),n_test)
    Y_test = np.sin(2*np.pi*np.array(X_test)) #计算出响应的sin（）函数值
    #向训练集合中的点的纵坐标加入噪声put noise into Y_train
    sigma = 0.1 #D(X) = 0
    mu = 0 #E(X) = 0
    for i in range(Y_train.size):
        Y_train[i] = Y_train[i] + random.gauss(mu,sigma)
    for i in range(Y_validate.size):
        Y_validate[i] = Y_validate[i] + random.gauss(mu,sigma)
    for i in range(Y_test.size):
        Y_test[i] = Y_test[i] + random.gauss(mu,sigma)

    return n_train,n_validate,n_test,X_train,Y_train,X_validate,Y_validate,X_test,Y_test


