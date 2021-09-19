# Task2回归问题

## 作业

### （1）请详细阐述线性回归模型的最小二乘法表达

​				基于均方误差最小化来进行模型求解的方法称为“最小二乘法”。在线性回归中，最小二乘法就是试图找到一条直线，使得所有样本到直线上的欧式距离之		和最小。
$$
L(w) = \sum\limits_{i=1}^{N}||w^Tx_i-y_i||_2^2=\sum\limits_{i=1}^{N}(w^Tx_i-y_i)^2 = (w^TX^T-Y^T)(w^TX^T-Y^T)^T = w^TX^TXw - 2w^TX^TY+YY^T\\
   因此，我们需要找到使得L(w)最小时对应的参数w，即：\\
   \hat{w} = argmin\;L(w)\\
   为了达到求解最小化L(w)问题，我们应用高等数学的知识，使用求导来解决这个问题： \\
   \frac{\partial L(w)}{\partial w} = 2X^TXw-2X^TY = 0,因此： \\
   \hat{w} = (X^TX)^{-1}X^TY
$$

### （2）在线性回归模型中，极大似然估计与最小二乘估计有什么联系与区别？

​				假设噪声$\epsilon \backsim N(0,\sigma^2),y=f(w)+\epsilon=w^Tx+\epsilon$

​				我们使用极大似然估计MLE对参数w进行估计：       
$$
L(w) = log\;P(Y|X;w) = log\;\prod_{i=1}^N P(y_i|x_i;w) = \sum\limits_{i=1}^{N} log\; P(y_i|x_i;w)\\
    = \sum\limits_{i=1}^{N}log(\frac{1}{\sqrt{2\pi \sigma}}exp(-\frac{(y_i-w^Tx_i)^2}{2\sigma^2})) = \sum\limits_{i=1}^{N}[log(\frac{1}{\sqrt{2\pi}\sigma})-\frac{1}{2\sigma^2}(y_i-w^Tx_i)^2] \\
    argmax_w L(w) = argmin_w[l(w) = \sum\limits_{i = 1}^{N}(y_i-w^Tx_i)^2]\\
    因此：线性回归的最小二乘估计<==>噪声\epsilon\backsim N(0,\sigma^2)的极大似然估计
$$

###  （3）为什么多项式回归在实际问题中的表现经常不是很好？  

​				为了体现因变量和特征的非线性关系，一个很自然而然的想法就是将标准的线性回归模型：             
$$
   y_i = w_0 + w_1x_i + \epsilon_i
$$
​				换成一个多项式函数：          
$$
   y_i = w_0 + w_1x_i + w_2x_i^2 + ...+w_dx_i^d + \epsilon
$$
​				对于多项式的阶数d不能取过大，一般不大于3或者4，因为d越大，多项式曲线就会越光滑，在X的边界处有异常的波动。（图中的边界处的4阶多项式拟		合曲线的置信区间(虚线表示置信区间)明显增大，预测效果的稳定性下降。）  

### （4）决策树模型与线性模型之间的联系与差别

​				线性模型的模型形式与树模型的模型形式有着本质的区别，具体而言，线性回归对模型形式做了如下假定：$f(x) = w_0 + \sum\limits_{j=1}^{p}w_jx^{(j)}$，而回归树则是$f(x) = \sum\limits_{m=1}^{J}\hat{c}_mI(x \in R_m)$。那问题来了，哪种模型更优呢？这个要视具体情况而言，如果特征变量与因变量的关系能很好的用线性关系来表达，那么线性回归通常有着不错的预测效果，拟合效果则优于不能揭示线性结构的回归树。反之，如果特征变量与因变量的关系呈现高度复杂的非线性，那么树方法比传统方法更优。

### （7）使用线性回归模型拟合模型

​				

```python
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

```

![](E:\Datawhale\HW\datawhale_ensemble-learning\Task02\LR.png)
