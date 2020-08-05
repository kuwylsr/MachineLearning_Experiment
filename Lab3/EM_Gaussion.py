import matplotlib.pyplot as plt
import xlrd
import numpy as np
import matplotlib as mpl
import math
from scipy.special import comb

#读取Excel文件
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
    x = filedata[:,:ncols-1]
    y = filedata[:,[ncols-1]]
    dict = {} #多维点映射为value值的字典
    for i in range(nrows-1):
        dict[str(x[i])] = y[i][0]
    x = np.array(x)
    return nrows-1,ncols-1,x,dict

"""
计算密度估计，以某个muj为期望，sigmaj为协方差的高斯分布在xi处的密度估计
"""
def DensityEstimation(listmu, listsigma, i,k):
    temp1 = 1 / (math.pow(math.pow(2*math.pi,2) * np.linalg.det(listsigma[k]), 0.5))
    temp2 = (-0.5 * np.mat(xGaussian[i] - listmu[k]))
    temp3 = np.linalg.inv(np.mat(listsigma[k]))
    temp4 = np.mat(xGaussian[i] - listmu[k]).T
    temp = np.exp((np.dot(np.dot(temp2,temp3),temp4)))
    return temp1 * temp

"""
计算高斯估计
"""
def GuassionEstimation(k):
    sum = 0
    for i in range(numN):
        sum = sum + W[i][k]
    return sum / numN

"""
更新mu参数
"""
def mean(k):
    sum1 = 0
    sum2 = 0
    for i in range(numN):
        sum1 = sum1 + W[i][k] * xGaussian[i]
        sum2 = sum2 + W[i][k]
    return sum1 / sum2

"""
更新sigma参数
"""
def variance(k):
    sum1 = 0
    sum2 = 0
    for i in range(numN):
        sum1 = sum1 + W[i][k] * (np.mat(xGaussian[i] - listmu[k])).T * (np.mat(xGaussian[i] - listmu[k]))
        sum2 = sum2 + W[i][k]
    return sum1 / sum2

"""
EM算法的E步骤
"""
def EPart():
    likelihood = 0
    for i in range(numN):
        sum = 0
        for k in range(numC):
            pxz = DensityEstimation(listmu,listsigma,i,k)
            W[i][k] = (pxz * listfai[k])
            sum = sum + pxz * listfai[k]
        W[i] = W[i] / sum
        likelihood = likelihood + sum
    print(likelihood)

"""
EM算法的M步骤
"""
def MPart():
    for k in range(numC):
        listfai[k] = GuassionEstimation(k)
        listmu[k] = mean(k)
        listsigma[k] = variance(k)


def EM():
    for i in range(10):
        # print(i)
        EPart()
        MPart()

def calculateARI():
    n = np.zeros([numC,numC])
    for i in range(numN):
        if(dict[str(xGaussian[i])] == 1):
            if(color[i] == 0):
                n[0][0] = n[0][0] + 1
            elif(color[i] == 1):
                n[0][1] = n[0][1] + 1
            elif(color[i] == 2):
                n[0][2] = n[0][2] + 1
        elif(dict[str(xGaussian[i])] == 2):
            if(color[i] == 0):
                n[1][0] = n[0][0] + 1
            elif(color[i] == 1):
                n[1][1] = n[0][1] + 1
            elif(color[i] == 2):
                n[1][2] = n[0][2] + 1
        elif(dict[str(xGaussian[i])] == 3):
            if(color[i] == 0):
                n[2][0] = n[0][0] + 1
            elif(color[i] == 1):
                n[2][1] = n[0][1] + 1
            elif(color[i] == 2):
                n[2][2] = n[0][2] + 1
    rowsum = n.sum(axis=1)
    colsum = n.sum(axis=0)

    s1 = 0
    s2 = 0
    s3 = 0
    for i in range(numC):
        s3 = 0
        s2 = s2 + comb(rowsum[i],2)
        for j in range(numC):
            s3 = s3 + comb(colsum[j],2)
            s1 = s1 + comb(n[i][j],2)
    ARI = (s1 - (s2 * s3)/comb(numN , 2)) / ((0.5 * (s2 + s3)) - (s2 * s3)/comb(numN,2))
    return ARI
"""
显示图像
"""
def paint(color):
    mpl.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
    #让生成的点均匀的分布在图中
    xmin = min(xGaussian[:, 0])
    xmax = max(xGaussian[:, 0])
    ymin = min(xGaussian[:, 1])
    ymax = max(xGaussian[:, 1])
    plt.xlim([xmin-3,xmax+3]) #限制横坐标的范围
    plt.ylim([ymin-3,ymax+3]) #限制纵坐标的范围
    ax = plt.subplot(211)
    ax.set_title("生成的高斯数据")
    ax.plot(xGaussian.T[0][:9],xGaussian.T[1][:9],'r.')
    ax.plot(xGaussian.T[0][:19][10:],xGaussian.T[1][:19][10:],'b.')
    ax.plot(xGaussian.T[0][20:],xGaussian.T[1][20:],'g.')
    bx = plt.subplot(212)
    bx.set_title("EM算法进行的聚类结果")
    bx.scatter(xGaussian.T[0], xGaussian.T[1], s=30, c = color)
    ARI = calculateARI()
    bx.text(1.5,0.5,"ARI:"+str(ARI))
    plt.show()

if __name__ == "__main__":
    nrows, ncols, xGaussian,dict = Readfile("MultiGaussianData.xls")

    numC = 3 #高斯分布的个数
    numN = nrows #样本点的个数
    numT = ncols #特征个数
    mu = np.empty([1,numT],dtype=float)
    sigma = np.zeros([numT,numT],dtype=float)
    W = np.empty([numN,numC],dtype=float)
    #初始化概率矩阵
    for i in range(numN):
        for k in range(numC):
            W[i][k] = (1/numC)

    listfai = []
    #先用k-means算法计算均值矩阵的初始值
    listmu = [[4.504775798341279, 4.318642177038526], [8.229822892210475, 8.235149129890766], [4.023293336114234, 10.137708613942943]]
    listsigma = []
    for k in range(numC):
    #初始化均值和协方差矩阵
        for j in range(numT):
            sigma[j][j] = 1
        listfai.append(1/3)
        listsigma.append(sigma)
    color = []
    EM()
    for i in range(numN):
        color.append((list)(W[i]).index(max(W[i])))

    paint(color)


