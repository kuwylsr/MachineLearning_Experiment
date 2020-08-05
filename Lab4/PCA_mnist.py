from ReadFileData import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #绘制3D坐标的函数
import numpy as np
import math

def txt_to_matrix(filename):
    file=open(filename)
    lines=file.readlines()
    #print lines
    #['0.94\t0.81\t...0.62\t\n', ... ,'0.92\t0.86\t...0.62\t\n']形式
    rows=len(lines)#文件行数

    datamat=np.zeros((rows,785))#初始化矩阵

    row=0
    for line in lines:
        line=line.strip().split(',')#strip()默认移除字符串首尾空格或换行符
        datamat[row,:]=line[:]
        row+=1

    return datamat

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

def CalculationSigma():
    temp = np.mat(x)
    sigma = np.zeros([numT,numT],dtype=int)
    for i in range(numN):
        sigma = sigma + np.dot(temp[i].T , temp[i])
    return sigma / numN

def PCA(numD,number):

    #Normalization()  #将数据正规化为零期望以及单位化方差
    sigma = CalculationSigma()
    a,b=np.linalg.eig(sigma)
    reduceDV = []
    dictFromValueToVector = {}
    for i in range(len(a.T)): #建立特征值到特征向量的映射
        dictFromValueToVector[str(a[i])] = np.array(b)[:,i]
    temp = sorted(a)
    temp.reverse()
    for j in range(numD):
        reduceDV.append(dictFromValueToVector[str(temp[j])])
    reduceDX = np.dot(np.mat(x[number]),np.mat(reduceDV).T)
    #重建
    rebuild = np.dot(reduceDX , reduceDV)
    return rebuild.real

def calculateSNR(number):
    origin = np.array(datamat[number,1:]).reshape(28,28)
    after = np.array(rebuild[0]).reshape(28,28)
    sum1 = 0
    sum2 = 0
    for i in range(28):
        for j in range(28):
            sum1 = sum1 + (origin[i][j] ** 2)
            sum2 = sum2 + ((origin[i][j] - after[i][j]) ** 2)
    SNR = 10 * math.log10(sum1 / sum2)
    return SNR

def paint(number):
    mpl.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
    SNR = calculateSNR(number)
    ax = plt.subplot(121)  # 创建一个三维的绘图工程
    ax.imshow(np.array(datamat[number,1:]).reshape(28,28),cmap=plt.cm.gray_r)

    bx = plt.subplot(122)
    bx.text(0,-1,"信噪比为："+str(SNR))
    bx.imshow(rebuild[0].reshape(28,28),cmap=plt.cm.gray_r)
    plt.show()

if __name__ == "__main__":
    # datamat = txt_to_matrix("MNIST_test.txt")
    datamat = txt_to_matrix("test5.txt")
    x = np.array(datamat)[:100,1:] #将第一列数字类别去掉
    numN = len(x)
    numT = 784
    rebuild = PCA(90,15)
    paint(15)




