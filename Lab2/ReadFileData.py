import xlrd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import random
import math

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
    return nrows,ncols,x,dict

