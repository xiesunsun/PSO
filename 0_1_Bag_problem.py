# 0-1背包问题。有N件物品和一个容量为V的背包。
# 第i件物品的体积是c（i），价值是w（i）。
# 求解将哪些物品放入背包可使物品的体积总和不超过背包的容量，且价值总和最大。
# 假设物品数量为10，背包的容量为300。
# 每件物品的体积为［95，75，23，73，50，22，6，57，89，98］
# 每件物品的价值为［89，59，19，43，100，72，44，16，7，64］。
import typing as t
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# 设置字体和设置负号
matplotlib.rc("font", family="KaiTi")
matplotlib.rcParams["axes.unicode_minus"] = False


# 已知条件
# volume = [95, 75, 23, 73, 50, 22, 6, 57, 89, 98]
# price = [89, 59, 19, 43, 100, 72, 44, 16, 7, 64]
# total_volume = 300
volume=[2,3,4,5]
price=[3,4,5,8]
total_volume=8
# 适应度函数
def value(x: t.List) -> t.Union[int, float]:
    c_volume = 0
    c_price = 0
    for i, j in enumerate(x):
        c_price += price[i] * j
        c_volume += volume[i] * j
    if c_volume > total_volume:
        return float("-inf")
    else:
        return c_price


# 粒子群算法的基本参数
N = 100  # 粒子个数
D = 4  # 空间维度
T = 10  # 迭代次数

# 个体学习因子
c1 = 1.5

# 社会学习因子
c2 = 1.5

# 惯性因子的参数
w_max = 0.8
w_min = 0.8

# 粒子飞行速度限制
v_max = 4
v_min = -4

# 初始化粒子的各项参数

# 初始化粒子的位置
x = np.random.randint(0, 2, [N, D])

# 初始化各个粒子的初始速度
v = (v_max - v_min) * np.random.rand(N, D) + v_min

#用于保存将速度转换成概率的矩阵
vx = np.random.rand(N, D)  

# 初始化每一个粒子的历史的最佳的位置
p = x


# 计算每个粒子当前位置的适应度是多少
p_best = np.ones(N) 
for i in range(N):
    p_best[i] = value(x[i, :])


# 初始化全局最优位置与最优解

#保存全局最优的适应度的数值
g_best = float("-inf")
# 达到全局最优适应度的粒子的位置
x_best = np.ones(D) 
for i in range(N):
    if p_best[i] > g_best:
        g_best = p_best[i]
        x_best = x[i, :].copy()



# 记录每次迭代的最优解
gb = np.ones(T)
for i in range(T):
    for j in range(N):
        if p_best[j] < value(x[j, :]):
            p_best[j] = value(x[j, :])
            p[j, :] = x[j, :].copy()
        # 更新全局最优位置与最优解
        if p_best[j] > g_best:
            g_best = p_best[j]
            x_best = x[j, :].copy()
        # 计算动态惯性权重
        w = w_max - (w_max - w_min) * i / T
        # print(type(v))
        # print(j)
        # print(v[j,:])
        
        v[j, :] = (
            w * v[j, :]
            + c1 * np.random.rand(1) * (p[j,:] - x[j, :])
            + c2 * np.random.rand(1) * (x_best - x[j, :])
        )
        
        
        
        # 边界条件处理
        for u in range(D):
            if (v[j, u] > v_max) or (v[j, u] < v_min):
                v[j, u] = v_min + np.random.rand(1) * (v_max - v_min)
        #进行概率计算并且更新相应的粒子的位置
        vx[j, :] = 1 / (1 + np.exp(-v[j, :]))
        for m in range(D):
            r = np.random.rand(1)
            x[j, m] = 1 if vx[j, m] > r else 0
    gb[i] = g_best
print("最优值为", gb[T - 1], "最优位置为", x_best)
plt.plot(range(T), gb)
plt.xlabel("迭代次数")

plt.ylabel("适应度值")
plt.title("适应度进化曲线")
plt.show()
