### Task 1 最小解发现

 

```
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def f(a, b, x, y):
    return (a - x) ** 2 + b * (y - x * x) ** 2


def plot_3d(a, b):
    # 定义x, y
    x = np.arange(-10, 10, 0.01)
    y = np.arange(-10, 10, 0.01)
    X, Y = np.meshgrid(x, y)  # 生成网格数据
    fig = plt.figure()  # 创建画布
    ax = Axes3D(fig)
    Z = f(a, b, X, Y)
    ax.plot_surface(X, Y, Z)  # 画图
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 图片中显示中文
    plt.title('Rosenbrock函数')
    plt.show()
```



2）编写一个算法来找到它的全局最小值及相应的最小解，并在3D图中标出。分析一下你的算法时空效率、给出运行时间。



梯度下降法求解

```
def f(a, b, x, y):
    return (a - x) ** 2 + b * (y - x * x) ** 2

#求解梯度向量
def grad(a, b, x, y):
    return np.array([2 * x - 2 * a + 4 * b * x * (x * x - y), 2 *b * (y - x * x)])

#梯度下降
def gradient_descent(a, b, grad, cur_x = np.array([0.1, 0.1]), learning_rate = 0.002, precision = 0.001, max_iters = 10000):
    for i in range(max_iters):
        grad_cur = grad(a, b, cur_x[0], cur_x[1])
        if np.linalg.norm(grad_cur, ord = 2) < precision:
            break
        cur_x = cur_x - grad_cur * learning_rate
        print("第", i, "次迭代：x 值为 ", cur_x)
    print("局部最小值 x =", cur_x)
    return cur_x
```

