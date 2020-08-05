from scipy.special import comb
import matplotlib.pyplot as plt
from random import choice
import xlrd
import numpy as np

MaxDistance = 100000

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
    return nrows,ncols-1,x,dict

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
    filedata = np.array(filedata)[:,:ncols-1]
    #将数据矩阵进行拆分，分为特征X矩阵和Y值矩阵
    filedata = np.array(filedata)
    return nrows-1,ncols-1,filedata

"""
初始化簇质心
"""
def initCC():
    ListC = []
    for i in range(numC):
        l = choice(x)
        ListC.append(list(l))
    return ListC

"""
更新每一个样本点，找到特们所属的簇质心
"""
def checkEveryNote():
    dictFromNtoC = {}
    for i in range(numN):
        minDistance = MaxDistance
        minCC = []
        for j in range(numC):
            distance = 0
            for k in range(numT):
                distance = distance + (x[i][k] - ListC[j][k]) **2
            if(distance < minDistance):
                minDistance = distance
                minCC = ListC[j]
        dictFromNtoC[str(x[i])] = minCC
    return dictFromNtoC


"""
更新簇质心
"""
def refreahCore(dictFromNtoC):
    for j in range(numC):
        l = [0 for i in range(numT)]
        temp2 = 0
        for i in range(numN):
            if(dictFromNtoC[str(x[i])] == ListC[j]):
                for k in range(numT):
                    l[k] += x[i][k]
                temp2 += 1
        for h in range(numT):
            l[h] = l[h] / temp2
        ListC[j] = l

"""
搜索某一个簇质心在下标
"""
def searchNum(mu):
    for j in range(numC):
        if(mu == ListC[j]):
            return j+1
    return 0

def KMeans():
    loss = 100000 #初始化失真函数代价
    while(1):
        temp = loss
        dictFromNtoC = checkEveryNote() #更新每一个样本点，找到它们所属的簇质心
        loss = calculateJ(dictFromNtoC) #计算更新后的失真函数代价
        paint(loss,dictFromNtoC) #画图
        refreahCore(dictFromNtoC) #更新簇质心
        if(abs(loss - temp) < 0.01):
            break

"""
显示图像
"""
def paint(loss,dictFromNtoC):

    #让生成的点均匀的分布在图中
    xmin = min(x[:,0])
    xmax = max(x[:,0])
    ymin = min(x[:,1])
    ymax = max(x[:,1])

    List = np.array(ListC)
    plt.clf()
    color = []
    for i in range(numN):
        Num = searchNum(dictFromNtoC[str(x[i])])
        color.append(Num)
    plt.xlim([xmin-3,xmax+3]) #限制横坐标的范围
    plt.ylim([ymin-3,ymax+3]) #限制纵坐标的范围
    plt.scatter(x.T[0],x.T[1],s=40,c = color)
    plt.plot(List.T[0],List.T[1],'rx')
    plt.text(xmin-3,ymax+4,'loss:'+str(loss))
    plt.pause(1)


def calculateARI(color):
    n = np.zeros([numC,numC])
    for i in range(numN):
        if(dict[str(x[i])] == 1):
            if(color[i] == 0):
                n[0][0] = n[0][0] + 1
            elif(color[i] == 1):
                n[0][1] = n[0][1] + 1
            elif(color[i] == 2):
                n[0][2] = n[0][2] + 1
        elif(dict[str(x[i])] == 2):
            if(color[i] == 0):
                n[1][0] = n[0][0] + 1
            elif(color[i] == 1):
                n[1][1] = n[0][1] + 1
            elif(color[i] == 2):
                n[1][2] = n[0][2] + 1
        elif(dict[str(x[i])] == 3):
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
计算失真函数的代价
"""
def calculateJ(dictFromNtoC):
    loss = 0
    for i in range(numN):
        tempx = (x[i][0] - dictFromNtoC[str(x[i])][0]) ** 2
        tempy = (x[i][1] - dictFromNtoC[str(x[i])][1]) ** 2
        loss += (tempx + tempy)
    return loss

if __name__ == "__main__":
    # nrows,ncols,x,dict = txt_to_matrix("dataUCI.txt")
    nrows,ncols,x = Readfile("MultiGaussianData.xls")

    numC = 3 #簇质心的个数
    numN = nrows #样本点的个数
    numT = ncols #特征数
    ListC = initCC()
    KMeans()
    plt.show()


