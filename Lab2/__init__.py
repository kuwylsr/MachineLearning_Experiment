

'''
MultiGaussian.py 文件来生成二维高斯分布的数据
GradientAscentN.py 文件使用梯度上升（不带惩罚项）(精度达到要求)求解最优参数，并且采用从Excel文件中读取生成的二维高斯分布进行分类
GradientAscentY.py 文件使用梯度上升（带有惩罚项）(精度达到要求)求解最优参数，对二维高斯分布进行分类
ReadFileData.py 从Excel文件读取样本数据
NewtonN.py 文件使用牛顿法（不带惩罚项）（固定迭代次数）来求解最优参数，并且从UCI网站上下载的Immunotherapy.xlsx文件进行分类来验证算法
NewtonN.py 文件使用牛顿法（带惩罚项）（固定迭代次数）来求解最优参数，并且从UCI网站上下载的Immunotherapy.xlsx文件进行分类来验证算法

源文件中共存在4个Excel文档：
    Immunotherapy.xlsx 从UCI网站上下载的关于冻伤的分类测试集
    MultiGaussianData.xls 二维高斯分布生成的满足logistic回归假设的测试集
    MultiGaussianDataAbnormal1.xls 二维高斯分布生成不满足Logistic回归第一个假设（朴素贝叶斯假设）的测试集
    MultiGaussianDataAbnormal2.xls 二维高斯分布生成不满足Logistic回归第二个假设的测试集
'''

__author__ = 'LiSirui 1160300610'
