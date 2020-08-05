import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math
from scipy.special import comb

"""
将txt文件的数据读进矩阵
"""
def txt_to_matrix(filename):
    file=open(filename)
    lines=file.readlines()
    #print lines
    #['0.94\t0.81\t...0.62\t\n', ... ,'0.92\t0.86\t...0.62\t\n']形式
    rows=len(lines)#文件行数
    datamat=np.zeros((rows,8))#初始化矩阵
    nrows=0
    for line in lines:
        line=line.strip().split('\t')#strip()默认移除字符串首尾空格或换行符
        datamat[nrows,:]=line[:]

        nrows+=1
    ncols = len(datamat.T)
    x = datamat[:,:ncols-1]
    y = datamat[:,[ncols-1]]
    dict = {} #多维点映射为value值的字典
    for i in range(nrows):
        dict[str(x[i])] = y[i][0]
    return nrows,ncols,x,dict

"""
计算密度估计，以某个muj为期望，sigmaj为协方差的高斯分布在xi处的密度估计
"""
def DensityEstimation(listmu, listsigma, i,k):
    temp1 = 1 / (math.pow(math.pow(2*math.pi,2) * np.linalg.det(listsigma[k]), 0.5))
    temp2 = (-0.5 * np.mat(xUCI[i] - listmu[k]))
    temp3 = np.linalg.inv(np.mat(listsigma[k]))
    temp4 = np.mat(xUCI[i] - listmu[k]).T
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
        sum1 = sum1 + W[i][k] * xUCI[i]
        sum2 = sum2 + W[i][k]
    return sum1 / sum2

"""
更新sigma参数
"""
def variance(k):
    sum1 = 0
    sum2 = 0
    for i in range(numN):
        sum1 = sum1 + W[i][k] * (np.mat(xUCI[i] - listmu[k])).T * (np.mat(xUCI[i] - listmu[k]))
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


"""
计算聚类问题的性能指标ARI
"""
def calculateARI():
    n = np.zeros([numC,numC])
    for i in range(numN):
        if(dict[str(xUCI[i])] == 1):
            if(color[i] == 0):
                n[0][0] = n[0][0] + 1
            elif(color[i] == 1):
                n[0][1] = n[0][1] + 1
            elif(color[i] == 2):
                n[0][2] = n[0][2] + 1
        elif(dict[str(xUCI[i])] == 2):
            if(color[i] == 0):
                n[1][0] = n[0][0] + 1
            elif(color[i] == 1):
                n[1][1] = n[0][1] + 1
            elif(color[i] == 2):
                n[1][2] = n[0][2] + 1
        elif(dict[str(xUCI[i])] == 3):
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

if __name__ == "__main__":
    nrows, ncols, xUCI , dict = txt_to_matrix("dataUCI.txt")

    numC = 3 #高斯分布的个数
    numN = nrows #样本点的个数
    numT = ncols-1 #特征个数
    mu = np.empty([1,numT],dtype=float)
    sigma = np.zeros([numT,numT],dtype=float)
    W = np.empty([numN,numC],dtype=float)
    #初始化概率矩阵
    for i in range(numN):
        for k in range(numC):
            W[i][k] = (1/numC)

    listfai = []
    #先用k-means算法计算均值矩阵的初始值
    listmu = [[14.819104477611939, 14.53716417910447, 0.8805223880597015, 5.591014925373135, 3.299358208955224, 2.706585074626865, 5.217537313432836], [11.988658536585366, 13.284390243902443, 0.852736585365854, 5.227426829268292, 2.880085365853659, 4.583926829268292, 5.074243902439023], [18.721803278688522, 16.297377049180326, 0.8850868852459014, 6.208934426229506, 3.7226721311475406, 3.603590163934426, 6.0660983606557375]]
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
    print("ARI = "+str(calculateARI()))


