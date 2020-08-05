
import numpy as np
import xlwt

#生成用于聚类分析的数类点

mu1 = [5,5] #均值矩阵
sigma1 = np.array([(1,0),(0,1)],dtype=int) #协方差矩阵
N1 = 10 #生成多维高斯分布的点的个数
X_Gaussian1 = [] #利用多维高斯函数生成的各维度列表的集合
m1 = len(X_Gaussian1) #维数
X_Gaussian1= np.random.multivariate_normal(mu1,sigma1,N1)
#============================================================
mu2 = [8,8] #均值矩阵
sigma2 = np.array([(1,0),(0,1)],dtype=int) #协方差矩阵
N2 = 10 #生成多维高斯分布的点的个数
X_Gaussian2 = [] #利用多维高斯函数生成的各维度列表的集合
m2 = len(X_Gaussian2) #维数
X_Gaussian2= np.random.multivariate_normal(mu2,sigma2,N2)
#============================================================
mu3 = [4,10] #均值矩阵
sigma3 = np.array([(1,0),(0,1)],dtype=int) #协方差矩阵
N3 = 10 #生成多维高斯分布的点的个数
X_Gaussian3 = [] #利用多维高斯函数生成的各维度列表的集合
m3 = len(X_Gaussian3) #维数
X_Gaussian3= np.random.multivariate_normal(mu3,sigma3,N3)

x = np.vstack((X_Gaussian1,X_Gaussian2,X_Gaussian3))

# #将多为高斯分布的数据存到excel表中（扩展名为：.xls）
# workbook = xlwt.Workbook(encoding= 'ascii')
# worksheet = workbook.add_sheet('MultiGaussianData')
# worksheet.write(0,0,label = 'x1')
# worksheet.write(0,1,label = 'x2')
# for i in range(N1 + N2 + N3):
#     for j in range(len(mu1)):
#         worksheet.write(i+1,j,label = x[i][j])
# workbook.save('MultiGaussianData.xls')


