
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #绘制3D坐标的函数
import numpy as np
import math
import xlrd

"""
读取文件
"""
def Readfile(name):
    data = xlrd.open_workbook(name) # 打开xls文件
    table = data.sheets()[0] # 打开第一张表
    nrows = table.nrows      # 获取表的行数
    ncols = table.ncols      # 获取表的列数
    filedata = [] #将表中的数据放入矩阵
    for i in range(nrows):   # 循环逐行打印
        if i == 0: # 跳过第一行
            continue
        filedata.append(table.row_values(i))

    #将数据矩阵进行拆分，分为特征X矩阵和Y值矩阵
    filedata = np.array(filedata)
    return nrows,ncols,filedata

"""
对数据样本进行预处理
"""
def Normalization():
    #使数据期望归零
    dataMu = 0
    for i in range(numN):
        dataMu = dataMu + x[i]
    for i in range(numN):
        x[i] = x[i] - dataMu
    #单位化方差
    for j in range(numT):
        dataSigma2 = 0
        for i in range(numN):
            dataSigma2 = dataSigma2 + math.pow(x[i][j] , 2)
        dataSigma2 = dataSigma2 / numN
        for i in range(numN):
            x[i][j] = x[i][j] / math.pow(dataSigma2 , 0.5)
"""
计算sigma矩阵
"""
def CalculationSigma():
    temp = np.mat(x)
    sigma = np.zeros([numT,numT],dtype=float)
    for i in range(numN):
        sigma = sigma + np.dot(temp[i].T , temp[i])
    return sigma / numN

def PCA(numD):

    #Normalization()  #将数据正规化为零期望以及单位化方差
    sigma = CalculationSigma()
    a,b=np.linalg.eig(sigma)

    reduceDV = []
    dictFromValueToVector = {}
    for i in range(len(a)): #建立特征值到特征向量的映射
        dictFromValueToVector[str(a[i])] = np.array(b)[:,i]
    temp = sorted(a)
    temp.reverse()
    for j in range(numD):
        reduceDV.append(dictFromValueToVector[str(temp[j])])
    reduceDX = np.dot(x,np.mat(reduceDV).T)
    return reduceDX

"""
画图
"""
def paint():
    ax = plt.subplot(211, projection='3d')  # 创建一个三维的绘图工程
    #让生成的点均匀的分布在图中
    xmin = min(x[:,0])
    xmax = max(x[:,0])
    ymin = min(x[:,1])
    ymax = max(x[:,1])
    zmin = min(x[:,2])
    zmax = max(x[:,2])
    ax.set_xlim([xmin-5,xmax+5])
    ax.set_ylim([ymin-5,ymax+5])
    ax.set_zlim([zmin-5,zmax+5])
    ax.plot(x.T[0],x.T[1],x.T[2],'r.')

    bx = plt.subplot(212)
    bx.plot(reduceDX.T[0],reduceDX.T[1],'r.')
    plt.show()

if __name__ == "__main__":
    nrows,ncols,x = Readfile("MultiGaussianData.xls")
    print(x)
    numN = len(x)
    numT = 3
    Normalization()
    print(x)
    # reduceDX = PCA(2)
    # paint()




