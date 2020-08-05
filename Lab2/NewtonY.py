from ReadFileData import *
from MultiGaussian import *

#使用牛顿法求解最优参数（带有惩罚项的）

#求解海森矩阵
def Hessian(X,W):
    A = np.zeros([n,n],dtype=float)
    for i in range(n):
        A[i][i] = h(X,W,i) * (h(X,W,i)-1)
    H = np.dot(np.dot(X.T,A),X)
    return H

#定义求h(sita)函数
def h(X,W,i):
    temp1 = 0
    for k in range(m-1):
        temp1 = temp1 + (W[k+1][0] * X[i][k+1])
    temp2 = np.exp(W[0][0] + temp1)
    temp = temp2 / (1 + temp2)
    return temp
    # return  1.0 / (1 + np.exp(np.dot(X[i] , W))) #矩阵形式

def Newton(X,Y,W):
    temp = np.empty([n,1],dtype=float)
    precision = 0.001 #精度
    max_iters = 1000 #固定迭代的次数
    iters = 0 #迭代的次数
    while (1):
        for i in range(n):
            temp[i][0] = Y[i] - h(X,W,i)
        U = np.dot(X.T,temp)
        # print(np.linalg.norm(Y-np.exp(np.dot(X, W))/(1 + np.exp(np.dot(X, W))), 2))
        W = W - (np.dot(np.linalg.pinv(Hessian(X,W)),U) - W * lam)/n #牛顿法（除以n，进行归一化处理）
        iters += 1
        if(iters % 100 == 0):
            print(iters)
        if iters == max_iters :
            print('reach the max iters! end!')
            break
    return W

#可视化分类结果
def paint():
    plt.xlim([0,18]) #限制横坐标的范围
    plt.ylim([0,18]) #限制纵坐标的范围
    x1 = x[:int(nrows/2),:]
    x2 = x[int(nrows/2)+1:,:]
    plt.plot(x1.T[0],x1.T[1],'b.')
    plt.plot(x2.T[0],x2.T[1],'r.')
    plt.plot(X_paint,Y_predict,'y-')
    accuracy,recall = index()
    plt.text(0,19,'准确率:'+str(accuracy))
    plt.text(0,20,'召回率:'+str(recall))
    plt.show()

#计算划分结果的策略指标
def index():
    Tp = 0. #分类正确且属于阳类别的个数
    Tn = 0. #分类正确且属于阴类别的个数
    N1 = 0. #测试样本中的Y=1（阳）类别的个数
    N2 = 0. #测试样本中Y=0（阴）类别的个数
    for i in range(n):
        temp = np.dot(X[i],W)
        if (dict[str(x[i])] == 1) : #样本属于阳例
            N1 = N1 + 1
            if (temp[0] > 0) : #分类出来样本属于阳例
                Tp = Tp + 1
        elif (dict[str(x[i])] == 0) : #样本属于阴例
            N2 = N2 + 1
            if (temp[0] < 0) : #分类出来样本属于阴例
                Tn = Tn + 1
    accuracy = (Tp + Tn) / (N1 + N2)
    recall = Tp / N1
    return accuracy , recall

if __name__ == "__main__":
    nrows,ncols,x,dict = Readfile("Immunotherapy.xlsx")
    mpl.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
    # m = len(mu1)+1 #特征数量-1
    # n = N1 + N2 #样本数量
    m = ncols #特征数量-1
    n = nrows-1 #样本数量
    X = np.empty([n,m],dtype=float)
    Y = np.empty([n,1],dtype=int)
    W = np.empty([m,1],dtype=float) #参数矩阵
    #初始化矩阵
    a = [[]] #将X矩阵最左侧加一列1
    for i in range(n):
        a[0].append(1)
    X = np.insert(x,0,values=a,axis=1)
    for i in range(n):
        Y[i][0] = dict[str(x[i])]
        for j in range(m):
            W[j][0] = 0
    #牛顿法求解最优解（带有惩罚项）
    lam = np.e ** (-10)
    W = Newton(X,Y,W)
    #生成用于绘制最优解函数的X坐标
    # X_paint = np.linspace(-5,5,1000)
    X_paint = np.linspace(0,20,1000)
    #构造以W为参数的降幂多项式函数
    func = np.poly1d([-(W[1][0]/W[2][0]),-(W[0][0]/W[2][0])])
    #利用函数求解X坐标对应的Y值
    Y_predict = func(X_paint)
    # paint();
    accuracy,recall = index()
    print('准确率:'+str(accuracy))
    print('召回率:'+str(recall))
