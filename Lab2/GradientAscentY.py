from MultiGaussian import *

#利用梯度上升法求解最优参数（带有惩罚项）

#定义求梯度的函数
def Gradient(X,Y,W,j):
    G = 0
    temp1 = 0
    for i in range(n):
        for k in range(m-1):
            temp1 = temp1 + (W[k+1][0] * X[i][k+1])
        temp2 = np.e ** (W[0][0] + temp1)
        temp = temp2 / (1 + temp2)
        G = G + X[i][j] * (Y[i] - temp)
        temp1 = 0

    return  (G - lam * W[j][0]) #加入惩罚项

#梯度上升求解最优解
def GradientDescent(X,Y,W):
    gamma = 0.01 #步长（学习率）
    precision = 0.05 #精度
    previous_step_size = []
    for j in range(m):
        previous_step_size.append(1)
    max_iters = 10000 #最多迭代的次数
    iters = 0 #迭代的次数
    flag = 1
    while (flag != 0):
        flag = 0
        for j in range(m):
            prev_x = W[j][0]
            W[j][0] = W[j][0] + (gamma * Gradient(X,Y,W,j)) #梯度上升
            previous_step_size[j] = abs(W[j][0] - prev_x)
        for j in range(m):
            if previous_step_size[j] > precision :
                flag = 1
        iters += 1
        if(iters % 100 == 0):
            print(iters)
        if iters > max_iters :
            print('reach the max iters! end!')
            break
    return W

#可视化分类结果
def paint():
    plt.xlim([0,18]) #限制横坐标的范围
    plt.ylim([0,18]) #限制纵坐标的范围
    plt.plot(X_Gaussian1.T[0],X_Gaussian1.T[1],'b.')
    plt.plot(X_Gaussian2.T[0],X_Gaussian2.T[1],'r.')
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
    mpl.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
    m = len(mu1)+1 #特征数量
    n = N1 + N2 #样本数量
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
    #用带有正则项的梯度上升法法求最优解
    lam = np.e ** (-10)
    W = GradientDescent(X,Y,W)
    #生成用于绘制最优解函数的X坐标
    X_paint = np.linspace(0,20,1000)
    #构造以W为参数的降幂多项式函数
    func = np.poly1d([-(W[1][0]/W[2][0]),-(W[0][0]/W[2][0])])
    #利用函数求解X坐标对应的Y值
    Y_predict = func(X_paint)
    paint()
