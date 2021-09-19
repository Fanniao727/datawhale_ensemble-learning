from numpy import *
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas


# 标准线性回归函数
def standRegres(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    # 判断行列式为零，则无法求逆
    if linalg.det(xTx) == 0:
        print('the matrix is singular, cannot do inverse')
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws


# 拟合数据
boston = datasets.load_boston()     # 返回一个类似于字典的类
xArr = boston.data
yArr = boston.target
features = boston.feature_names
boston_data = pandas.DataFrame(xArr,columns=features)
boston_data["Price"] = yArr
boston_data.head()

ws = standRegres(xArr, yArr)
xMat = mat(xArr)
yMat = mat(yArr)
yHat = xMat * ws
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
xCopy = xMat.copy()
xCopy.sort(0)
yHat = xCopy * ws
ax.plot(xCopy[:, 1], yHat)
plt.show()
