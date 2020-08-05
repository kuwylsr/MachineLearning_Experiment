from GaussianNoise import *

#共轭梯度的优化算法来求解最优解
def Conjgrad(A,b,X):
    r = b - np.dot(A,X)
    p = r
    rsold = np.dot(r.T,r)
    for i in range(len(b)):
        Ap = np.dot(A,p)
        alpha = rsold / (np.dot(p.T,Ap))
        X = X + alpha * p
        r = r - alpha * Ap
        rsnew = np.dot(r.T,r)
        if np.sqrt(rsnew) < 1e-10:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return X

def paintCurve(pic):
    #计算解析解的代价（有正则项）
    Y_fromTrain = func(X_train) #通过拟合出来的曲线函数计算训练样本中X对应的Y值
    E_train = 0
    Y_fromTest = func(X_test) #通过拟合出来的曲线函数计算测试样本中X对应的Y值
    E_test = 0
    for i in range(n_train):
        E_train += (Y_fromTrain[i] - Y_train[i]) ** 2 #计算对训练样本的拟合程度的代价
    for i in range(n_test):
        E_test += (Y_fromTest[i] - Y_test[i]) ** 2 #计算对测试样本拟合程度的代价
    E_train = E_train * 0.5 * (1/n_train)
    E_test = E_test * 0.5 * (1/n_test)
    w2 = 0
    for i in range(m):
        w2 = w2 + (W[i][0] * W[i][0])
    E_train = (E_train +(lam/2)*w2)/n_train
    E_test = (E_test +(lam/2)*w2)/n_test
    plt.subplot(331+pic) #在一个画板上生成多幅子图
    plt.xlim([-0.15,1.15]) #限制横坐标的范围
    plt.ylim([-1.25,1.25]) #限制纵坐标的范围
    plt.plot(X_train,Y_train,'b.',X_paint,Y_predict,'r-') #画出训练样本的点和拟合出的曲线
    # plt.text(-0.1,1.27,'$(n)loss:$'+str(E_train)) #展示loss
    # plt.text(-0.1,1.35,'$(n)loss:$'+str(E_test)) #展示loss
    plt.text(0,-0.4,'$m:$'+str(m-1)) #展示多项式函数的阶数
    plt.text(0,-0.6,'$n:$'+str(n_train)) #展示训练样本的数量
    loss_training_x.append(m-1)
    loss_training_y.append(E_train)
    loss_testing_x.append(m-1)
    loss_testing_y.append(E_test)

def paintloss():
    plt.subplot(313)
    plt.ylim([0,0.2]) #限制纵坐标的范围
    plt.title("训练样本代价与测试样本代价",fontsize=16)
    plt.xlabel("m")
    plt.plot(loss_training_x,loss_training_y,'bx-')
    plt.plot(loss_testing_x,loss_testing_y,'ro-')
    plt.show()

if __name__ == "__main__" :
    mpl.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
    plt.figure(figsize=[12,8])
    N = 40 #总的样本数量
    n_train,n_validate,n_test,X_train,Y_train,X_validate,Y_validate,X_test,Y_test = ProductGN(N)
    loss_training_x = [] #针对训练样本的代价列表
    loss_training_y = []
    loss_testing_x = [] #针对测试样本的代价列表
    loss_testing_y = []
    loss_validate_x = [] #针对验证样本的代价列表
    loss_validate_y = []
    pic = 0
    for num in [1,2,4,6,8,10]:
        m = num
        X = np.empty([n_train,m],dtype=float) #定义输入变量矩阵
        T = np.empty([n_train,1],dtype=float) #定义输出变量矩阵
        W = np.empty([m,1],dtype=float) #定义参数矩阵
        #初始化X，T矩阵
        for i in range(n_train):
            T[i][0] = (Y_train[i])
            for j in range(m):
                X[i][j] = (X_train[i])**j
                W[j][0]=0
        #用带有正则项的解析法求最优解
        lam = np.e ** (-30)
        W = Conjgrad(np.dot(X.T,X)+lam,np.dot(X.T,T),W)
        #生成用于绘制最优解函数的X坐标
        X_paint = np.linspace(0,1,1000)
        W1 = W[::-1].reshape(m)
        #构造以W为参数的降幂多项式函数
        func = np.poly1d(W1)
        #利用函数求解X坐标对应的Y值
        Y_predict = func(X_paint)
        paintCurve(pic)
        pic += 1
    paintloss()
