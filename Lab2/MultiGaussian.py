import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import matplotlib as mpl
import numpy as np
import random
import math
import xlwt

#生成两类服从高斯分布的数据
#当协方差矩阵的有对角线数值不为0时，不满足同一类别的特征之间不独立。（第一个假设）（朴素贝叶斯假设）
#当两个高斯分布的协方差矩阵不对应相等的化，不满足不同类别之间同一特征不相同。（第二个假设）

sigma1 = np.array([(5,0),(0,5)],dtype=int) #协方差矩阵
mu1 = [5,5] #均值矩阵
N1 = 100 #生成多维高斯分布的点的个数
X_Gaussian1 = [] #利用多维高斯函数生成的各维度列表的集合
dict = {} #多维点映射为value值的字典
m1 = len(X_Gaussian1) #维数
x1 = np.empty([N1,m1]) #点的列表

X_Gaussian1= np.random.multivariate_normal(mu1,sigma1,N1)
x1 = X_Gaussian1
for i in range(N1):
    dict[str(x1[i])] = 0
#============================================================

mu2 = [10,10] #均值矩阵
sigma2 = np.array([(5,0),(0,5)],dtype=int) #协方差矩阵
N2 = 100 #生成多维高斯分布的点的个数
X_Gaussian2 = [] #利用多维高斯函数生成的各维度列表的集合
m2 = len(X_Gaussian2) #维数
x2 = np.empty([N2,m2]) #点的列表

X_Gaussian2= np.random.multivariate_normal(mu2,sigma2,N2)
x2 = X_Gaussian2
for i in range(N2):
    dict[str(x2[i])] = 1

x = np.vstack((x1,x2))

#将多为高斯分布的数据存到excel表中（扩展名为：.xls）
workbook = xlwt.Workbook(encoding= 'ascii')
worksheet = workbook.add_sheet('MultiGaussianData')
worksheet.write(0,0,label = 'x1')
worksheet.write(0,1,label = 'x2')
worksheet.write(0,2,label = 'Y')
for i in range(N1 + N2):
    for j in range(len(mu1)+1):
        if(j != 2):
            worksheet.write(i+1,j,label = x[i][j])
        else:
            worksheet.write(i+1,j,label = dict[str(x[i])])
workbook.save('MultiGaussianData.xls')


