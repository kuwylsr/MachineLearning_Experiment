import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #绘制3D坐标的函数
import numpy as np
import xlwt
import random

#生成三维高斯分布的样本数据

mu = [10,10,10] #均值矩阵
sigma = np.array([(1000,0,0),(0,1000,0),(0,0,1)],dtype=int) #协方差矩阵
N1 = 20 #生成多维高斯分布的点的个数
X_Gaussian1 = [] #利用多维高斯函数生成的各维度列表的集合
m1 = len(X_Gaussian1) #维数
X_Gaussian1= np.random.multivariate_normal(mu,sigma,N1)

#将多为高斯分布的数据存到excel表中（扩展名为：.xls）
workbook = xlwt.Workbook(encoding= 'ascii')
worksheet = workbook.add_sheet('MultiGaussianData')
worksheet.write(0,0,label = 'x1')
worksheet.write(0,1,label = 'x2')
worksheet.write(0,2,label = 'x3')
for i in range(N1):
    for j in range(len(mu)):
        worksheet.write(i+1,j,label = X_Gaussian1[i][j])
workbook.save('MultiGaussianData.xls')




